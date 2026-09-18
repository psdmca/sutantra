#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Universal PDF to Tamil / Unicode Text Extractor with Tesseract OCR Support
=========================================================================
A fast, self-contained Python script (zero external library dependencies) that:
1. Parses PDF binary objects and decompresses zlib streams.
2. Dynamically traverses the PDF /Pages tree across any document.
3. Automatically decodes embedded TrueType/OpenType fonts (cmap Format 4,
   GSUB Lookup Type 4 Ligature Substitutions) and PDF /ToUnicode CMaps.
4. Correctly unescapes PDF literal strings and parses TJ/Tj operators.
5. Reorders visual-encoded Tamil vowels (ெ, ே, ை) into Unicode phonetic order.
6. Clusters text chunks spatially into cohesive lines and stanzas.
7. Supports Scanned Document Pages via Tesseract OCR:
   - Natively extracts embedded raster images (CCITT Group 4 Fax TIFF, JPEG, PBM).
   - Runs Tesseract OCR with Tamil language support (--ocr, --ocr-lang tam+eng).
   - Supports page range selection (--ocr-pages 1-10) and image export (--save-images).
   - Gracefully detects missing OCR dependencies and guides installation.

Usage:
    python3 tools/pdf_to_txt_extractor.py <input_pdf> [output_txt] [options]
"""

import sys
import os
import re
import zlib
import struct
import time
import shutil
import tempfile
import subprocess
import argparse


def parse_pdf_objects(pdf_bytes):
    """Builds a fast byte-offset index of all PDF objects."""
    obj_positions = {}
    for m in re.finditer(rb'\b(\d+)\s+0\s+obj\b', pdf_bytes):
        obj_positions[int(m.group(1))] = m.start()
    return obj_positions


def get_obj_data(pdf_bytes, obj_positions, num):
    """Returns raw bytes of an object dictionary and definition."""
    pos = obj_positions.get(num)
    if pos is None:
        return b''
    end_pos = pdf_bytes.find(b'endobj', pos)
    if end_pos == -1:
        return b''
    return pdf_bytes[pos:end_pos]


def get_obj_stream(pdf_bytes, obj_positions, num):
    """Decompresses and returns the stream data of an object."""
    data = get_obj_data(pdf_bytes, obj_positions, num)
    sm = re.search(rb'stream\r?\n(.*?)\r?\nendstream', data, re.DOTALL)
    if not sm:
        return None
    raw = sm.group(1)
    try:
        return zlib.decompress(raw)
    except Exception:
        return raw


def find_pages_root(pdf_bytes, obj_positions):
    """Dynamically locates the root /Pages object in any PDF."""
    trail_m = re.findall(rb'/Root\s+(\d+)\s+0\s+R', pdf_bytes)
    if trail_m:
        cat_data = get_obj_data(pdf_bytes, obj_positions, int(trail_m[-1]))
        pages_m = re.search(rb'/Pages\s+(\d+)\s+0\s+R', cat_data)
        if pages_m:
            return int(pages_m.group(1))

    # Fallback: search for root /Pages object with no /Parent
    for num in obj_positions:
        data = get_obj_data(pdf_bytes, obj_positions, num)
        if (b'/Type/Pages' in data or b'/Type /Pages' in data) and b'/Parent' not in data:
            return num
    return None


def get_page_objects(pdf_bytes, obj_positions, pages_root_num):
    """Recursively collects all /Page object numbers from the /Pages tree."""
    pages = []
    visited = set()

    def collect(node_num):
        if node_num in visited:
            return
        visited.add(node_num)
        data = get_obj_data(pdf_bytes, obj_positions, node_num)
        # Check if Page node
        if (b'/Type/Page' in data or b'/Type /Page' in data) and (b'/Type/Pages' not in data and b'/Type /Pages' not in data):
            pages.append(node_num)
            return
        # Internal /Pages node with /Kids
        kids_m = re.search(rb'/Kids\s*\[(.*?)\]', data, re.DOTALL)
        if kids_m:
            kids = [int(x) for x in re.findall(rb'(\d+)\s+0\s+R', kids_m.group(1))]
            for k in kids:
                collect(k)

    if pages_root_num:
        collect(pages_root_num)
    return pages


def parse_tounicode(cmap_bytes):
    """Parses PDF /ToUnicode CMap stream into a dictionary of cid -> Unicode text."""
    cmap = {}
    text = cmap_bytes.decode('latin1', errors='ignore')

    def hex_to_str(h):
        return ''.join(chr(int(h[i:i+4], 16)) for i in range(0, len(h), 4))

    # 1. beginbfchar / endbfchar
    for section in re.finditer(r'beginbfchar(.*?)endbfchar', text, re.DOTALL):
        for line in section.group(1).strip().splitlines():
            m = re.findall(r'<([0-9A-Fa-f]+)>', line)
            if len(m) >= 2:
                src = int(m[0], 16)
                cmap[src] = hex_to_str(m[1])

    # 2. beginbfrange / endbfrange
    for section in re.finditer(r'beginbfrange(.*?)endbfrange', text, re.DOTALL):
        for line in section.group(1).strip().splitlines():
            line = line.strip()
            # Array form: <start> <end> [ <dst1> <dst2> ... ]
            m_arr = re.match(r'<([0-9A-Fa-f]+)>\s+<([0-9A-Fa-f]+)>\s+\[(.*?)\]', line)
            if m_arr:
                start = int(m_arr.group(1), 16)
                dsts = re.findall(r'<([0-9A-Fa-f]+)>', m_arr.group(3))
                for idx, dst_hex in enumerate(dsts):
                    cmap[start + idx] = hex_to_str(dst_hex)
                continue
            # Consecutive form: <start> <end> <base_dst>
            m_consec = re.match(r'<([0-9A-Fa-f]+)>\s+<([0-9A-Fa-f]+)>\s+<([0-9A-Fa-f]+)>', line)
            if m_consec:
                start = int(m_consec.group(1), 16)
                end = int(m_consec.group(2), 16)
                dst_hex = m_consec.group(3)
                if len(dst_hex) == 4:
                    base = int(dst_hex, 16)
                    for i in range(end - start + 1):
                        cmap[start + i] = chr(base + i)
                else:
                    cmap[start] = hex_to_str(dst_hex)
    return cmap


def build_font_decoder_from_ttf(ttf_data):
    """
    Parses TrueType / OpenType font binary data:
    - Extracts cmap table (subtable 4, Unicode BMP)
    - Extracts GSUB table (Lookup Type 4, Ligature Substitution)
    - Maps all GIDs (Glyph IDs) to correct Tamil Unicode strings
    """
    if not ttf_data or len(ttf_data) < 12:
        return {}

    num_tables = struct.unpack('>H', ttf_data[4:6])[0]
    tables = {}
    for i in range(num_tables):
        tag, check, offset, length = struct.unpack('>4sIII', ttf_data[12+i*16:12+(i+1)*16])
        tables[tag.decode('latin1')] = (offset, length)

    gid_to_char = {}

    # 1. Parse cmap table
    if 'cmap' in tables:
        cmap_offset, cmap_len = tables['cmap']
        ver, num_subtables = struct.unpack('>HH', ttf_data[cmap_offset:cmap_offset+4])
        for i in range(num_subtables):
            plat, enc, sub_off = struct.unpack('>HHI', ttf_data[cmap_offset+4+i*8:cmap_offset+12+i*8])
            data = ttf_data[cmap_offset+sub_off:]
            fmt = struct.unpack('>H', data[:2])[0]
            if fmt == 4:
                length, lang, seg_count_x2 = struct.unpack('>HHH', data[2:8])
                seg_count = seg_count_x2 // 2
                end_codes = struct.unpack(f'>{seg_count}H', data[14:14+seg_count*2])
                start_codes = struct.unpack(f'>{seg_count}H', data[16+seg_count*2:16+seg_count*4])
                id_deltas = struct.unpack(f'>{seg_count}h', data[16+seg_count*4:16+seg_count*6])
                id_range_offsets = struct.unpack(f'>{seg_count}H', data[16+seg_count*6:16+seg_count*8])
                for s in range(seg_count - 1):
                    for code in range(start_codes[s], end_codes[s] + 1):
                        if id_range_offsets[s] == 0:
                            gid = (code + id_deltas[s]) & 0xFFFF
                        else:
                            loc = 16 + seg_count*6 + s*2
                            off = loc + id_range_offsets[s] + (code - start_codes[s])*2
                            if off + 2 <= len(data):
                                gid = struct.unpack('>H', data[off:off+2])[0]
                                if gid != 0:
                                    gid = (gid + id_deltas[s]) & 0xFFFF
                            else:
                                gid = 0
                        if gid != 0 and gid not in gid_to_char:
                            gid_to_char[gid] = chr(code)

    # Combining marks in GIST Indic fonts
    if 0x95 in gid_to_char and gid_to_char[0x95] == '\u0BB5':
        gid_to_char[0x97] = '\u0BB8' # ஸ
        gid_to_char[0x98] = '\u0BB9' # ஹ
        gid_to_char[0x9A] = '\u0BBF' # ி
        gid_to_char[0x9B] = '\u0BC0' # ீ
        gid_to_char[0x9C] = '\u0BC1' # ு
        gid_to_char[0x9D] = '\u0BC2' # ூ
        gid_to_char[0xA4] = '\u0BCD' # ்

    # 2. Parse GSUB table
    if 'GSUB' in tables:
        def parse_coverage(s_data, cov_off):
            cov_fmt = struct.unpack('>H', s_data[cov_off:cov_off+2])[0]
            glyphs = []
            if cov_fmt == 1:
                cnt = struct.unpack('>H', s_data[cov_off+2:cov_off+4])[0]
                glyphs = list(struct.unpack(f'>{cnt}H', s_data[cov_off+4:cov_off+4+cnt*2]))
            elif cov_fmt == 2:
                rcnt = struct.unpack('>H', s_data[cov_off+2:cov_off+4])[0]
                for r in range(rcnt):
                    s, e, si = struct.unpack('>HHH', s_data[cov_off+4+r*6:cov_off+10+r*6])
                    glyphs.extend(range(s, e + 1))
            return glyphs

        gsub_offset, gsub_len = tables['GSUB']
        gdata = ttf_data[gsub_offset:gsub_offset+gsub_len]
        ver_maj, ver_min, script_off, feat_off, look_off = struct.unpack('>HHHHH', gdata[:10])
        look_count = struct.unpack('>H', gdata[look_off:look_off+2])[0]
        ligatures = {}
        for l_idx in range(look_count):
            l_off = struct.unpack('>H', gdata[look_off+2+l_idx*2:look_off+4+l_idx*2])[0]
            l_type, l_flag, sub_cnt = struct.unpack('>HHH', gdata[look_off+l_off:look_off+l_off+6])
            if l_type != 4:  # Type 4: Ligature Substitution
                continue
            for s_idx in range(sub_cnt):
                sub_off = struct.unpack('>H', gdata[look_off+l_off+6+s_idx*2:look_off+l_off+8+s_idx*2])[0]
                s_data = gdata[look_off+l_off+sub_off:]
                fmt, cov_off, lig_set_cnt = struct.unpack('>HHH', s_data[:6])
                cov_glyphs = parse_coverage(s_data, cov_off)
                lig_set_offsets = struct.unpack(f'>{lig_set_cnt}H', s_data[6:6+lig_set_cnt*2])
                for first_gid, ls_off in zip(cov_glyphs, lig_set_offsets):
                    ls_data = s_data[ls_off:]
                    lig_cnt = struct.unpack('>H', ls_data[:2])[0]
                    lig_offsets = struct.unpack(f'>{lig_cnt}H', ls_data[2:2+lig_cnt*2])
                    for lig_off in lig_offsets:
                        lig_data = ls_data[lig_off:]
                        lig_glyph, comp_cnt = struct.unpack('>HH', lig_data[:4])
                        comps = [first_gid]
                        if comp_cnt > 1:
                            comps.extend(struct.unpack(f'>{comp_cnt-1}H', lig_data[4:4+(comp_cnt-1)*2]))
                        ligatures[lig_glyph] = comps

        changed = True
        while changed:
            changed = False
            for lig_gid, comp_gids in ligatures.items():
                if lig_gid not in gid_to_char:
                    if all(c in gid_to_char for c in comp_gids):
                        gid_to_char[lig_gid] = ''.join(gid_to_char[c] for c in comp_gids)
                        changed = True

    return gid_to_char


# Precompiled regex patterns for visual-to-phonetic Tamil reordering
C_PATTERN = r'(?:[\u0B95-\u0BB9](?:\u0BCD[\u0B95-\u0BB9])?)'
P_O_SHORT = re.compile(r'\u0BC6(' + C_PATTERN + r')\u0BBE')  # ெ + C + ா -> C + ொ
P_AU      = re.compile(r'\u0BC6(' + C_PATTERN + r')\u0BD7')  # ெ + C + ௗ -> C + ௌ
P_O_LONG  = re.compile(r'\u0BC7(' + C_PATTERN + r')\u0BBE')  # ே + C + ா -> C + ோ
P_E_SHORT = re.compile(r'\u0BC6(' + C_PATTERN + r')')        # ெ + C     -> C + ெ
P_E_LONG  = re.compile(r'\u0BC7(' + C_PATTERN + r')')        # ே + C     -> C + ே
P_AI      = re.compile(r'\u0BC8(' + C_PATTERN + r')')        # ை + C     -> C + ை


def reorder_tamil_visual(text):
    """Converts visual-order pre-base Tamil vowels to Unicode phonetic order."""
    text = P_O_SHORT.sub(lambda m: m.group(1) + '\u0BCA', text)
    text = P_AU.sub(lambda m: m.group(1) + '\u0BCC', text)
    text = P_O_LONG.sub(lambda m: m.group(1) + '\u0BCB', text)
    text = P_E_SHORT.sub(lambda m: m.group(1) + '\u0BC6', text)
    text = P_E_LONG.sub(lambda m: m.group(1) + '\u0BC7', text)
    text = P_AI.sub(lambda m: m.group(1) + '\u0BC8', text)
    return text


def parse_tj_array(tj_str):
    """
    Parses a PDF TJ array string properly unescaping literal strings.
    Handles hex strings <...>, escaped literal strings (...), and kerning numbers.
    """
    tokens = []
    i = 0
    n = len(tj_str)
    while i < n:
        c = tj_str[i]
        if c == '<':
            j = tj_str.find('>', i)
            if j != -1:
                tokens.append(('hex', tj_str[i+1:j].replace(' ', '')))
                i = j + 1
            else:
                break
        elif c == '(':
            j = i + 1
            paren_depth = 1
            s_chars = []
            while j < n and paren_depth > 0:
                if tj_str[j] == '\\' and j + 1 < n:
                    esc = tj_str[j+1]
                    if esc == 'n':
                        s_chars.append('\n')
                    elif esc == 'r':
                        s_chars.append('\r')
                    elif esc == 't':
                        s_chars.append('\t')
                    elif esc in ('(', ')', '\\'):
                        s_chars.append(esc)
                    else:
                        s_chars.append(esc)
                    j += 2
                elif tj_str[j] == '(':
                    paren_depth += 1
                    s_chars.append('(')
                    j += 1
                elif tj_str[j] == ')':
                    paren_depth -= 1
                    if paren_depth > 0:
                        s_chars.append(')')
                    j += 1
                else:
                    s_chars.append(tj_str[j])
                    j += 1
            tokens.append(('str', ''.join(s_chars)))
            i = j
        elif c in '-0123456789':
            j = i
            while j < n and (tj_str[j] in '-0123456789.'):
                j += 1
            try:
                tokens.append(('num', float(tj_str[i:j])))
            except ValueError:
                pass
            i = j
        else:
            i += 1
    return tokens


def clean_tamil_line(line):
    """Cleans up spaces and punctuation in a single line."""
    line = line.strip()
    if not line:
        return ''
    line = re.sub(r'[ \t]+', ' ', line)
    # Remove space before punctuation: , . ; : ! ? ) ”
    line = re.sub(r'\s+([,.;:!?\)\”])', r'\1', line)
    # Ensure space after comma/semicolon/colon if followed by word character
    line = re.sub(r'([,;:!?])([^\s\d\)\”])', r'\1 \2', line)
    # Ensure space before opening parenthesis / quote if preceded by non-space
    line = re.sub(r'([^\s\(\“])([\(\“])', r'\1 \2', line)
    return line


def get_font_file2_for_font(pdf_bytes, obj_positions, f_num):
    """Finds the FontFile2 (TTF stream) object number for a font."""
    data = get_obj_data(pdf_bytes, obj_positions, f_num)
    fd = re.search(rb'/FontDescriptor\s+(\d+)\s+0\s+R', data)
    if fd:
        fd_data = get_obj_data(pdf_bytes, obj_positions, int(fd.group(1)))
        ff2 = re.search(rb'/FontFile2\s+(\d+)\s+0\s+R', fd_data)
        if ff2:
            return int(ff2.group(1))

    desc = re.search(rb'/DescendantFonts\s*(?:\[\s*(\d+)|\s+(\d+))\s+0\s+R', data)
    if desc:
        d_num = int(desc.group(1) or desc.group(2))
        d_data = get_obj_data(pdf_bytes, obj_positions, d_num)
        cid_match = re.search(rb'(\d+)\s+0\s+R', d_data)
        if b'/FontDescriptor' not in d_data and cid_match:
            d_data = get_obj_data(pdf_bytes, obj_positions, int(cid_match.group(1)))
        fd = re.search(rb'/FontDescriptor\s+(\d+)\s+0\s+R', d_data)
        if fd:
            fd_data = get_obj_data(pdf_bytes, obj_positions, int(fd.group(1)))
            ff2 = re.search(rb'/FontFile2\s+(\d+)\s+0\s+R', fd_data)
            if ff2:
                return int(ff2.group(1))
    return None


def build_font_decoder_for_font(pdf_bytes, obj_positions, f_num, font_cache):
    """Dynamically builds and caches a character decoder for any PDF font object."""
    if f_num in font_cache:
        return font_cache[f_num]

    decoder = {}
    f_data = get_obj_data(pdf_bytes, obj_positions, f_num)

    # 1. Base decoding from embedded TrueType cmap & GSUB (most authoritative for ligatures)
    ff2_num = get_font_file2_for_font(pdf_bytes, obj_positions, f_num)
    if ff2_num:
        ttf_data = get_obj_stream(pdf_bytes, obj_positions, ff2_num)
        if ttf_data:
            try:
                ttf_dec = build_font_decoder_from_ttf(ttf_data)
                decoder.update(ttf_dec)
            except Exception:
                pass

    # 2. Augment from /ToUnicode CMap
    tu_m = re.search(rb'/ToUnicode\s+(\d+)\s+0\s+R', f_data)
    if not tu_m:
        desc = re.search(rb'/DescendantFonts\s*(?:\[\s*(\d+)|\s+(\d+))\s+0\s+R', f_data)
        if desc:
            d_num = int(desc.group(1) or desc.group(2))
            d_data = get_obj_data(pdf_bytes, obj_positions, d_num)
            tu_m = re.search(rb'/ToUnicode\s+(\d+)\s+0\s+R', d_data)

    if tu_m:
        tu_data = get_obj_stream(pdf_bytes, obj_positions, int(tu_m.group(1)))
        if tu_data:
            try:
                tu_map = parse_tounicode(tu_data)
                for cid, uchar in tu_map.items():
                    if cid not in decoder or not decoder[cid].strip():
                        decoder[cid] = uchar
            except Exception:
                pass

    # 3. Standard printable ASCII fallback
    for c in range(32, 127):
        if c not in decoder:
            decoder[c] = chr(c)

    font_cache[f_num] = decoder
    return decoder


def get_fonts_for_page(pdf_bytes, obj_positions, page_num):
    """Resolves all font aliases for a given page, including inherited resources."""
    curr_num = page_num
    while curr_num:
        data = get_obj_data(pdf_bytes, obj_positions, curr_num)
        # Direct /Font << ... >>
        f_in = re.search(rb'/Font\s*<<(.*?)>>', data, re.DOTALL)
        if f_in:
            res = {}
            for m in re.finditer(rb'/([A-Za-z0-9_-]+)\s+(\d+)\s+0\s+R', f_in.group(1)):
                res[m.group(1).decode('latin1')] = int(m.group(2))
            if res:
                return res
        # Indirect /Font N 0 R
        f_ref = re.search(rb'/Font\s+(\d+)\s+0\s+R', data)
        if f_ref:
            f_data = get_obj_data(pdf_bytes, obj_positions, int(f_ref.group(1)))
            f_inner = re.search(rb'<<(.*?)>>', f_data, re.DOTALL)
            if f_inner:
                res = {}
                for m in re.finditer(rb'/([A-Za-z0-9_-]+)\s+(\d+)\s+0\s+R', f_inner.group(1)):
                    res[m.group(1).decode('latin1')] = int(m.group(2))
                if res:
                    return res
        # Walk up /Parent
        parent_m = re.search(rb'/Parent\s+(\d+)\s+0\s+R', data)
        curr_num = int(parent_m.group(1)) if parent_m else None
    return {}


def get_page_contents(pdf_bytes, obj_positions, page_num):
    """Retrieves all concatenated content stream bytes for a page."""
    data = get_obj_data(pdf_bytes, obj_positions, page_num)
    c_m = re.search(rb'/Contents\s*\[(.*?)\]', data, re.DOTALL)
    if c_m:
        c_nums = [int(x) for x in re.findall(rb'(\d+)\s+0\s+R', c_m.group(1))]
    else:
        single_m = re.search(rb'/Contents\s+(\d+)\s+0\s+R', data)
        c_nums = [int(single_m.group(1))] if single_m else []

    parts = []
    for num in c_nums:
        st = get_obj_stream(pdf_bytes, obj_positions, num)
        if st:
            parts.append(st)
    return b'\n'.join(parts)


def build_g4_tiff(raw_bytes, width, height):
    """
    Constructs a standard, valid Little-Endian CCITT Group 4 Fax TIFF image
    directly from raw CCITTFaxDecode stream bytes without external libraries.
    """
    tags = [
        (256, 4, 1, width),            # ImageWidth
        (257, 4, 1, height),           # ImageLength
        (258, 3, 1, 1),                # BitsPerSample
        (259, 3, 1, 4),                # Compression: CCITT Group 4 Fax
        (262, 3, 1, 0),                # PhotometricInterpretation: 0 (WhiteIsZero)
        (273, 4, 1, 8 + 2 + 9*12 + 4), # StripOffsets (122)
        (277, 3, 1, 1),                # SamplesPerPixel
        (278, 4, 1, height),           # RowsPerStrip
        (279, 4, 1, len(raw_bytes)),   # StripByteCounts
    ]
    ifd = struct.pack('<H', len(tags))
    for tag, t_type, count, val in tags:
        if t_type == 3:  # SHORT
            ifd += struct.pack('<HHIHH', tag, t_type, count, val, 0)
        else:            # LONG
            ifd += struct.pack('<HHII', tag, t_type, count, val)
    ifd += struct.pack('<I', 0)  # Next IFD offset
    header = struct.pack('<2sHI', b'II', 42, 8) + ifd
    return header + raw_bytes


def get_page_image_info(pdf_bytes, obj_positions, page_num):
    """
    Extracts the primary scanned raster image for a page.
    Returns (image_bytes, file_extension, width, height) or None.
    """
    data = get_obj_data(pdf_bytes, obj_positions, page_num)

    def extract_from_dict(d):
        nums = []
        xo_m = re.search(rb'/XObject\s*<<(.*?)>>', d, re.DOTALL)
        if xo_m:
            for m in re.finditer(rb'/([A-Za-z0-9_-]+)\s+(\d+)\s+0\s+R', xo_m.group(1)):
                nums.append(int(m.group(2)))
        xo_ref = re.search(rb'/XObject\s+(\d+)\s+0\s+R', d)
        if xo_ref:
            rd = get_obj_data(pdf_bytes, obj_positions, int(xo_ref.group(1)))
            for m in re.finditer(rb'/([A-Za-z0-9_-]+)\s+(\d+)\s+0\s+R', rd):
                nums.append(int(m.group(2)))
        return nums

    xobj_nums = extract_from_dict(data)
    res_m = re.search(rb'/Resources\s+(\d+)\s+0\s+R', data)
    if res_m:
        xobj_nums.extend(extract_from_dict(get_obj_data(pdf_bytes, obj_positions, int(res_m.group(1)))))

    img_num = None
    for num in xobj_nums:
        obj_d = get_obj_data(pdf_bytes, obj_positions, num)
        if b'/Subtype/Image' in obj_d or b'/Subtype /Image' in obj_d:
            img_num = num
            break
        if b'/Subtype/Form' in obj_d or b'/Subtype /Form' in obj_d:
            for inum in extract_from_dict(obj_d):
                in_d = get_obj_data(pdf_bytes, obj_positions, inum)
                if b'/Subtype/Image' in in_d or b'/Subtype /Image' in in_d:
                    img_num = inum
                    break
        if img_num:
            break

    if not img_num:
        return None

    img_data = get_obj_data(pdf_bytes, obj_positions, img_num)
    sm = re.search(rb'stream\r?\n(.*?)\r?\nendstream', img_data, re.DOTALL)
    if not sm:
        return None
    raw_stream = sm.group(1)

    w_m = re.search(rb'/Width\s+(\d+)', img_data)
    h_m = re.search(rb'/Height\s+(\d+)', img_data)
    width = int(w_m.group(1)) if w_m else 0
    height = int(h_m.group(1)) if h_m else 0

    if b'/CCITTFaxDecode' in img_data:
        tiff_data = build_g4_tiff(raw_stream, width, height)
        return (tiff_data, '.tif', width, height)
    elif b'/DCTDecode' in img_data:
        return (raw_stream, '.jpg', width, height)
    elif b'/FlateDecode' in img_data:
        try:
            decomp = zlib.decompress(raw_stream)
        except Exception:
            decomp = raw_stream
        pbm_header = f'P4\n{width} {height}\n'.encode('ascii')
        return (pbm_header + decomp, '.pbm', width, height)

    return (raw_stream, '.bin', width, height)


def run_tesseract_ocr(image_bytes, image_ext, lang='tam+eng', tesseract_cmd='tesseract'):
    """
    Executes Tesseract OCR on in-memory image bytes.
    Returns recognized text string or None if Tesseract is not found.
    """
    tess_path = shutil.which(tesseract_cmd)
    if not tess_path:
        return None

    with tempfile.NamedTemporaryFile(suffix=image_ext, delete=False) as tmp_in:
        tmp_in_path = tmp_in.name
        tmp_in.write(image_bytes)

    tmp_out_base = tmp_in_path + "_ocr"
    try:
        cmd = [tess_path, tmp_in_path, tmp_out_base, '-l', lang]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if proc.returncode != 0:
            return ""
        out_txt = tmp_out_base + ".txt"
        if os.path.exists(out_txt):
            with open(out_txt, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
            return text
        return ""
    finally:
        if os.path.exists(tmp_in_path):
            try:
                os.remove(tmp_in_path)
            except Exception:
                pass
        out_txt = tmp_out_base + ".txt"
        if os.path.exists(out_txt):
            try:
                os.remove(out_txt)
            except Exception:
                pass


def parse_page_range(range_str, total_pages):
    """Parses page ranges such as '1-10', '1,3,5', or 'all' into 0-indexed page list."""
    if not range_str or range_str.strip().lower() == 'all':
        return list(range(total_pages))
    pages = set()
    for part in range_str.split(','):
        part = part.strip()
        if '-' in part:
            s, e = part.split('-', 1)
            start_idx = max(0, int(s) - 1)
            end_idx = min(total_pages, int(e))
            pages.update(range(start_idx, end_idx))
        elif part.isdigit():
            idx = int(part) - 1
            if 0 <= idx < total_pages:
                pages.add(idx)
    return sorted(pages)


def extract_pdf_to_text(pdf_path, output_txt_path=None, enable_ocr=False,
                        ocr_lang='tam+eng', ocr_pages=None, save_images_dir=None):
    """
    Universal PDF to text extractor function with Tesseract OCR support.
    1. Extracts digital text streams and decodes fonts automatically.
    2. If the PDF contains scanned pages, triggers Tesseract OCR or guides setup.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Input PDF not found: {pdf_path}")

    output_txt_requested = (output_txt_path is not None)
    if not output_txt_path:
        base, _ = os.path.splitext(pdf_path)
        output_txt_path = base + ".txt"

    print(f"Opening PDF: {pdf_path}")
    file_size_mb = os.path.getsize(pdf_path) / (1024 * 1024)
    print(f"File size: {file_size_mb:.2f} MB")
    t0 = time.time()

    with open(pdf_path, 'rb') as f:
        pdf_bytes = f.read()

    obj_positions = parse_pdf_objects(pdf_bytes)
    print(f"Indexed {len(obj_positions)} PDF objects in {time.time() - t0:.2f}s")

    pages_root = find_pages_root(pdf_bytes, obj_positions)
    if not pages_root:
        raise ValueError("Could not locate root /Pages tree in this PDF")

    page_objs = get_page_objects(pdf_bytes, obj_positions, pages_root)
    total_pages = len(page_objs)
    print(f"Found {total_pages} pages in document tree")

    if total_pages == 0:
        print("[WARNING] Document contains 0 pages.")
        return ""

    # Check if user requested image export
    if save_images_dir:
        os.makedirs(save_images_dir, exist_ok=True)
        target_img_indices = parse_page_range(ocr_pages, total_pages)
        print(f"Exporting {len(target_img_indices)} page images to: {save_images_dir}...")
        exported = 0
        for idx in target_img_indices:
            p_num = page_objs[idx]
            img_info = get_page_image_info(pdf_bytes, obj_positions, p_num)
            if img_info:
                b_data, ext, w, h = img_info
                img_path = os.path.join(save_images_dir, f"page_{idx+1:04d}{ext}")
                with open(img_path, 'wb') as img_f:
                    img_f.write(b_data)
                exported += 1
        print(f"Successfully exported {exported} page images to {save_images_dir}")

    # Digital text extraction pass
    font_cache = {}
    all_pages_lines = []
    pages_with_text = 0
    t_ext_start = time.time()

    if not enable_ocr:
        for page_idx, page_obj_num in enumerate(page_objs):
            f_map = get_fonts_for_page(pdf_bytes, obj_positions, page_obj_num)
            c_bytes = get_page_contents(pdf_bytes, obj_positions, page_obj_num)
            if not c_bytes:
                all_pages_lines.append([])
                continue

            c_text = c_bytes.decode('latin1', errors='ignore')
            items = []
            for m in re.finditer(r'BT\s*(.*?)\s*ET', c_text, re.DOTALL):
                b_text = m.group(1)
                tm_m = re.search(r'([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)\s+Tm', b_text)
                x = float(tm_m.group(5)) if tm_m else 0.0
                y = float(tm_m.group(6)) if tm_m else 0.0

                f_m = re.search(r'(/F[A-Za-z0-9_-]+)\s+[\d.]+\s+Tf', b_text)
                f_alias = f_m.group(1).replace('/', '') if f_m else ''
                f_num = f_map.get(f_alias)
                dec = build_font_decoder_for_font(pdf_bytes, obj_positions, f_num, font_cache) if f_num else {}

                t_ops = re.findall(r'\[(.*?)\]\s*TJ|\((.*?)\)\s*Tj', b_text)
                for tj_arr, tj_str in t_ops:
                    if tj_arr:
                        tokens = parse_tj_array(tj_arr)
                        txt = ''
                        for t_type, t_val in tokens:
                            if t_type == 'hex':
                                for i in range(0, len(t_val), 4):
                                    code = int(t_val[i:i+4], 16)
                                    txt += dec.get(code, '')
                            elif t_type == 'str':
                                txt += t_val
                            elif t_type == 'num' and t_val < -100:
                                txt += ' '
                        if txt:
                            items.append((y, x, txt))
                    elif tj_str:
                        s_clean = tj_str.replace(r'\(', '(').replace(r'\)', ')').replace(r'\\', '\\')
                        if s_clean:
                            items.append((y, x, s_clean))

            if items:
                pages_with_text += 1

            lines_dict = {}
            for y, x, txt in items:
                found_key = None
                for lk in lines_dict:
                    if abs(lk - y) < 3.0:
                        found_key = lk
                        break
                if found_key is None:
                    found_key = y
                    lines_dict[found_key] = []
                lines_dict[found_key].append((x, txt))

            sorted_y = sorted(lines_dict.keys(), reverse=True)
            page_lines = []
            prev_y = None

            for y in sorted_y:
                if prev_y is not None and (prev_y - y) > 23.0:
                    if page_lines and page_lines[-1] != '':
                        page_lines.append('')

                row = lines_dict[y]
                row.sort(key=lambda item: item[0])
                line_txt = ''.join(t for _, t in row)
                line_txt = reorder_tamil_visual(line_txt)
                line_txt = clean_tamil_line(line_txt)
                if line_txt:
                    if re.match(r'^\d+$', line_txt) and (y < 80.0 or y > 730.0):
                        continue
                    page_lines.append(line_txt)
                    prev_y = y

            all_pages_lines.append(page_lines)

        print(f"Processed {total_pages} pages in {time.time() - t_ext_start:.2f}s ({pages_with_text} pages contained text operators)")

    # Check if OCR is needed
    if pages_with_text == 0 or enable_ocr:
        tess_installed = shutil.which('tesseract') is not None

        if not tess_installed:
            print("\n" + "="*74)
            print("[NOTICE: SCANNED IMAGE DOCUMENT]")
            print(f"This PDF contains {total_pages} scanned image pages and no digital text layer.")
            print("\nTesseract OCR executable is not installed or not found in PATH.")
            print("To enable automatic Tamil OCR extraction, install Tesseract:")
            print("    sudo apt update && sudo apt install -y tesseract-ocr tesseract-ocr-tam")
            print("\nOnce installed, run:")
            print(f"    python3 {sys.argv[0]} \"{pdf_path}\" \"{output_txt_path}\" --ocr")
            print("\nYou can also extract and inspect the high-resolution scanned page images:")
            print(f"    python3 {sys.argv[0]} \"{pdf_path}\" --save-images ./scanned_pages/")
            print("="*74 + "\n")

            if not output_txt_requested:
                print("[INFO] No text file generated because document contains no digital text layer.")
                print(f"Completed in {time.time() - t0:.2f}s.")
                return ""

        else:
            # Tesseract is available! Run OCR
            target_indices = parse_page_range(ocr_pages, total_pages)
            print(f"\nRunning Tesseract OCR ({ocr_lang}) on {len(target_indices)} pages...")
            t_ocr_start = time.time()
            all_pages_lines = []

            for progress_idx, p_idx in enumerate(target_indices):
                page_obj = page_objs[p_idx]
                img_info = get_page_image_info(pdf_bytes, obj_positions, page_obj)
                if not img_info:
                    continue

                b_data, ext, w, h = img_info
                print(f"  [{progress_idx+1}/{len(target_indices)}] OCR Page {p_idx+1}/{total_pages} ({w}x{h} {ext})...", end='', flush=True)
                t_p0 = time.time()
                raw_ocr_text = run_tesseract_ocr(b_data, ext, lang=ocr_lang)
                t_dur = time.time() - t_p0
                print(f" done ({t_dur:.1f}s)")

                p_lines = []
                for l in (raw_ocr_text or '').splitlines():
                    cleaned = clean_tamil_line(l)
                    if cleaned:
                        p_lines.append(cleaned)
                all_pages_lines.append(p_lines)

            print(f"Completed OCR on {len(target_indices)} pages in {time.time() - t_ocr_start:.2f}s")

    # Combine lines into final document
    book_lines = []
    for plines in all_pages_lines:
        if not plines:
            continue
        for l in plines:
            book_lines.append(l)
        book_lines.append("")

    final_text = '\n'.join(book_lines)
    final_text = re.sub(r'\n{3,}', '\n\n', final_text).strip()
    if final_text:
        final_text += '\n'

    if not final_text:
        if output_txt_requested:
            out_dir = os.path.dirname(output_txt_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            with open(output_txt_path, 'w', encoding='utf-8') as f:
                f.write('')
            print(f"Empty output written to: {output_txt_path}")
        else:
            print("[INFO] No text file generated because document contains no digital text layer.")
        print(f"Completed in {time.time() - t0:.2f}s.")
        return ""

    # Ensure output directory exists
    out_dir = os.path.dirname(output_txt_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print(f"Writing output ({len(final_text)} characters, {len(final_text.splitlines())} lines) to: {output_txt_path}")
    with open(output_txt_path, 'w', encoding='utf-8') as f:
        f.write(final_text)

    print(f"Done in {time.time() - t0:.2f}s!")
    return final_text


def main():
    parser = argparse.ArgumentParser(
        description="Universal, zero-dependency PDF to Tamil / Unicode text extractor with Tesseract OCR support."
    )
    parser.add_argument("input_pdf", help="Path to the input PDF file")
    parser.add_argument("output_txt", nargs="?", default=None, help="Path to the output text file (optional)")
    parser.add_argument("--ocr", action="store_true", help="Enable Tesseract OCR for scanned document pages")
    parser.add_argument("--ocr-lang", default="tam+eng", help="Tesseract language code (default: 'tam+eng')")
    parser.add_argument("--ocr-pages", default=None, help="Page range to OCR (e.g. '1-10', '1,2,5', default: all)")
    parser.add_argument("--save-images", default=None, help="Directory to export scanned page images (.tif, .jpg, .pbm)")

    args = parser.parse_args()
    extract_pdf_to_text(
        pdf_path=args.input_pdf,
        output_txt_path=args.output_txt,
        enable_ocr=args.ocr,
        ocr_lang=args.ocr_lang,
        ocr_pages=args.ocr_pages,
        save_images_dir=args.save_images
    )


if __name__ == '__main__':
    main()
