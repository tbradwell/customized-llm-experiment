#!/usr/bin/env python3
"""
FODT-based document editing pipeline:
  DOCX -> FODT (LibreOffice) -> collect text nodes -> LLM (per slice) -> update FODT -> DOCX

Why FODT over HTML?
- Better layout/formatting preservation 
- Native LibreOffice format for more faithful conversion
- Granular text node editing without structural changes

Requirements:
  pip install openai beautifulsoup4 lxml
  LibreOffice installed (soffice in PATH)
  export OPENAI_API_KEY=...

Usage: edit the CONFIG paths at the bottom and run.
"""

import os, subprocess, shutil, tempfile
from pathlib import Path
from typing import List, Tuple
from bs4 import BeautifulSoup, NavigableString, Tag
from openai import OpenAI

from concurrent.futures import ThreadPoolExecutor

from src.utils.rag import FODTTextMatcher

CONCURRENCY = 3
MAX_RETRIES = 3
BACKOFF_BASE_S = 0.8


def load_new_data(new_data_dir: str) -> str:
    """Load and combine new data from directory."""
    import sys
    sys.path.append('/home/tzuf/Desktop/projects/customized-LLM-experiments')
    from src.processors.data_loader import DataLoader
    
    data_loader = DataLoader()
    return data_loader.load_new_data(new_data_dir, is_flat_text=False)

def collect_text_nodes_from_fodt(soup: BeautifulSoup) -> List[Tuple[NavigableString, str]]:
    """
    Walk FODT XML and collect text nodes without touching structure.
    FODT uses office:document-content -> office:body -> office:text
    Text content is in <text:p>, <text:span>, etc. elements.
    """
    text_nodes = []
    
    def walk_node(node, context_path=""):
        if isinstance(node, NavigableString):
            text_content = str(node).strip()
            # Only collect meaningful text content (skip whitespace-only)
            if text_content and len(text_content) > 2:
                text_nodes.append((node, f"{context_path}: '{text_content[:50]}'"))
        elif isinstance(node, Tag):
            # Build context showing ODT structure
            tag_info = node.name
            if node.get('text:style-name'):
                tag_info += f"[{node.get('text:style-name')}]"
            
            new_context = f"{context_path}/{tag_info}" if context_path else tag_info
            
            # Walk all children
            for child in node.children:
                walk_node(child, new_context)
    
    # Start from document root
    walk_node(soup)
    return text_nodes

def group_text_nodes_into_slices(text_nodes: List[Tuple[NavigableString, str]], max_tokens: int = 800) -> List[List[Tuple[NavigableString, str]]]:
    """
    Group text nodes into slices for batch LLM processing.
    Keep related content together while staying under token limit.
    """
    if not text_nodes:
        return []
    
    slices = []
    current_slice = []
    current_tokens = 0
    
    for text_node, context in text_nodes:
        text_content = str(text_node)
        # Rough token estimate: ~4 chars per token
        tokens = max(1, len(text_content) // 4)
        
        # If adding this node would exceed limit, start new slice
        if current_slice and current_tokens + tokens > max_tokens:
            slices.append(current_slice)
            current_slice = []
            current_tokens = 0
        
        current_slice.append((text_node, context))
        current_tokens += tokens
    
    # Add final slice if not empty
    if current_slice:
        slices.append(current_slice)
    
    return slices

def edit_text_node_slice_with_llm(
    text_slice: List[Tuple[NavigableString, str]], 
    client, model: str, user_instruction: str, new_data: str) -> None:
    """
    Send a slice of text nodes to LLM for editing and replace in-place.
    Preserves exact FODT XML structure and formatting.
    """
    system_prompt = """You are a legal document editor specializing in updating case information while preserving document style. CRITICAL RULES:

FORMATTING PRESERVATION:
- Edit ONLY the text content, NEVER touch XML structure, tags, or attributes
- Preserve ALL whitespace, unicode bidi marks, RTL/LTR direction markers  
- Keep Hebrew/English mixed text layout exactly as provided
- Never add or remove XML elements, namespaces, or formatting tags
- Maintain exact character encoding and special characters

CONTENT UPDATING APPROACH:
- UPDATE factual content (names, dates, amounts, case details) with new data
- PRESERVE the original writing style, tone, and sentence structure
- KEEP the same legal phrasing patterns and terminology
- MAINTAIN the same level of formality and document flow
- Only change what needs to be updated based on new data
- If new data doesn't relate to a text segment, leave it completely unchanged"""
    
    if not text_slice:
        return
    
    # Build context for LLM showing all text in this slice
    texts_to_edit = []
    for i, (text_node, context) in enumerate(text_slice):
        original_text = str(text_node).strip()
        if len(original_text) >= 3:  # Only include meaningful text
            texts_to_edit.append(f"{i+1}. {original_text}")
    
    if not texts_to_edit:
        return
    
    user_prompt = f"""{user_instruction}

NEW DATA TO INCORPORATE:
{new_data}

CURRENT TEXT SEGMENTS TO UPDATE:
{chr(10).join(texts_to_edit)}

TASK: Update the text segments above by incorporating relevant information from the NEW DATA. 
- Maintain the exact same numbered format (1. 2. 3. etc.)
- Keep segments that don't relate to the new data unchanged
- For relevant segments, update with new data while preserving the original style and legal phrasing
- Ensure Hebrew/English mixed text layout remains intact
- Only change the content/facts, preserve the writing style, tone, and structure
- Return all segments in the same order, even if unchanged"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
        )
        
        edited_content = response.choices[0].message.content.strip()
        
        # Parse response and update text nodes
        edited_lines = [line.strip() for line in edited_content.split('\n') if line.strip()]
        
        text_index = 0
        for line in edited_lines:
            if line and '. ' in line:
                try:
                    # Extract number and text: "1. edited text here"
                    num_str, edited_text = line.split('. ', 1)
                    num = int(num_str) - 1  # Convert to 0-based index
                    
                    if 0 <= num < len(text_slice):
                        text_node, _ = text_slice[num]
                        original_text = str(text_node)
                        
                        # Only update if actually changed
                        if edited_text != original_text.strip():
                            # Preserve original whitespace structure
                            if original_text.startswith(' '):
                                edited_text = ' ' + edited_text
                            if original_text.endswith(' '):
                                edited_text = edited_text + ' '
                            
                            text_node.replace_with(edited_text)
                            print(f"Updated: '{original_text[:50]}...' -> '{edited_text[:50]}...'")
                
                except (ValueError, IndexError) as e:
                    print(f"Warning: Could not parse LLM response line '{line[:50]}...': {e}")
                    continue
                    
    except Exception as e:
        print(f"Warning: LLM editing failed for slice: {e}")
        # Keep all original text on error

# -----------------------
# Conversion helpers
# -----------------------

def run(cmd: List[str]) -> None:
    print(f"Running command: {' '.join(cmd)}")
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(f"Command output: {p.stdout}")
    print(f"Return code: {p.returncode}")
    if p.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{p.stdout}")

def soffice_docx_to_fodt(docx_path: Path, out_dir: Path) -> Path:
    """Convert DOCX to FODT (Flat ODT) format which preserves layout better than HTML."""
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Converting DOCX to FODT: {docx_path} -> {out_dir}")
    print(f"DOCX file exists: {docx_path.exists()}")
    
    run(["soffice", "--headless", "--convert-to", "fodt",
         "--outdir", str(out_dir), str(docx_path)])
    
    expected_name = docx_path.with_suffix(".fodt").name
    produced = out_dir / expected_name
    print(f"Expected FODT file: {produced}")
    print(f"Expected file exists: {produced.exists()}")
    
    if not produced.exists():
        print(f"Expected file not found. Checking directory: {out_dir}")
        all_files = list(out_dir.glob("*"))
        print(f"Files in directory: {[f.name for f in all_files]}")
        
        # fallback to any .fodt generated
        cands = list(out_dir.glob("*.fodt"))
        if not cands:
            raise FileNotFoundError(f"LibreOffice did not produce FODT. Directory contents: {[f.name for f in all_files]}")
        produced = cands[0]
        print(f"Using actual FODT file: {produced}")
    
    return produced

def soffice_fodt_to_docx(fodt_path: Path, out_docx: Path) -> None:
    """Convert FODT (Flat ODT) back to DOCX format."""
    out_docx.parent.mkdir(parents=True, exist_ok=True)
    print(f"Converting FODT to DOCX: {fodt_path} -> {out_docx}")
    print(f"FODT file exists: {fodt_path.exists()}")
    if fodt_path.exists():
        print(f"FODT file size: {fodt_path.stat().st_size} bytes")
    
    run(["soffice", "--headless", "--convert-to", "docx:Office Open XML Text",
         "--outdir", str(out_docx.parent), str(fodt_path)])
    
    produced = out_docx.parent / (fodt_path.stem + ".docx")
    print(f"Looking for produced file: {produced}")
    print(f"Produced file exists: {produced.exists()}")
    
    if not produced.exists():
        print(f"Expected file not found. Checking directory: {out_docx.parent}")
        all_files = list(out_docx.parent.glob("*"))
        print(f"Files in directory: {[f.name for f in all_files]}")
        
        cands = list(out_docx.parent.glob("*.docx"))
        if not cands:
            raise FileNotFoundError(f"LibreOffice did not produce DOCX. Directory contents: {[f.name for f in all_files]}")
        produced = cands[0]
        print(f"Using alternate DOCX file: {produced}")
    
    if produced != out_docx:
        print(f"Moving {produced} to {out_docx}")
        if out_docx.exists():
            out_docx.unlink()
        produced.replace(out_docx)

# Removed unused HTML slicing and reassembly functions since this is now a FODT-focused pipeline

# -----------------------
# Main
# -----------------------

def main():
    # ====== CONFIG ======

    INPUT_DOCX = Path("/home/tzuf/Desktop/projects/customized-LLM-experiments/examples/amit_test2/small_test.docx")          # <-- change me
    NEW_DATA_DIR = Path("/home/tzuf/Desktop/projects/customized-LLM-experiments/examples/amit_test2/new_data/")
    WORK_DIR = Path(tempfile.mkdtemp(prefix="docx_html_slice_"))
    OUTPUT_DOCX = WORK_DIR / "output.docx"
    MODEL = "gpt-5"
    # MODEL = os.getenv("OPENAI_MODEL", "gpt-5")
    SYSTEM_PROMPT = "You are a careful HTML copy editor. You only edit text; you never change markup."
    USER_INSTRUCTION = "Update the legal document with new case information while preserving the original document structure, formatting, and legal language style."
    USE_PARALLEL_PROCESSING = True  # Set to True for parallel, False for sequential
    # ====================

    try:
        if not shutil.which("soffice"):
            raise EnvironmentError("LibreOffice 'soffice' not found on PATH.")
        if not INPUT_DOCX.exists():
            raise FileNotFoundError(f"Input not found: {INPUT_DOCX}")

        # 1) DOCX -> FODT (Flat ODT preserves layout better than HTML)
        fodt_path = soffice_docx_to_fodt(INPUT_DOCX, WORK_DIR)
        fodt_content = fodt_path.read_text(encoding="utf-8", errors="ignore")

        # 2) Parse FODT XML with lxml-xml (proper XML parser for ODT format)
        soup = BeautifulSoup(fodt_content, "lxml-xml")

        # 3) Collect text nodes for editing (preserve exact XML structure)
        text_nodes = collect_text_nodes_from_fodt(soup)
        print(f"Found {len(text_nodes)} text nodes to potentially edit")

        # 4) Group text nodes into slices for batch processing
        text_slices = group_text_nodes_into_slices(text_nodes, max_tokens=800)
        print(f"Grouped into {len(text_slices)} slices for LLM processing")

        # get new data
        matcher = FODTTextMatcher(
            os.getenv("OPENAI_API_KEY"), 'text-embedding-3-small')
        list_new_data = matcher._load_new_data(NEW_DATA_DIR)
        matches = matcher.find_best_matches(text_slices[0], list_new_data, top_k=2)
        new_data = '\n\n'.join([m['text'] for m in matches])

        # 5) Edit each slice with LLM (with proper OpenAI integration)
        if text_slices:
            try:
                from openai import OpenAI
                client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                
                if USE_PARALLEL_PROCESSING:
                    print("Using parallel processing for LLM editing...")
                    # Process slices in parallel for faster editing
                    with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
                        futures = []
                        for i, text_slice in enumerate(text_slices):
                            print(f"Submitting slice {i+1}/{len(text_slices)} ({len(text_slice)} text nodes)...")
                            future = executor.submit(edit_text_node_slice_with_llm, text_slice, client, MODEL, USER_INSTRUCTION, new_data)
                            futures.append((i+1, future))
                        
                        # Wait for all slices to complete
                        for slice_num, future in futures:
                            try:
                                future.result()  # This will raise any exceptions from the worker
                                print(f"Completed slice {slice_num}/{len(text_slices)}")
                            except Exception as e:
                                print(f"Error processing slice {slice_num}: {e}")
                else:
                    print("Using sequential processing for LLM editing...")
                    # Process slices sequentially for better debugging/control
                    for i, text_slice in enumerate(text_slices):
                        try:
                            print(f"Processing slice {i+1}/{len(text_slices)} ({len(text_slice)} text nodes)...")
                            edit_text_node_slice_with_llm(text_slice, client, MODEL, USER_INSTRUCTION, new_data)
                            print(f"Completed slice {i+1}/{len(text_slices)}")
                        except Exception as e:
                            print(f"Error processing slice {i+1}: {e}")
                            print("Continuing with next slice...")
                    
            except Exception as e:
                print(f"Warning: LLM processing failed: {e}")
                print("Continuing with original text content...")

        # 6) Serialize back to FODT XML (preserves structure exactly)
        final_fodt = str(soup)
        final_fodt_path = WORK_DIR / "final_edited.fodt"
        final_fodt_path.write_text(final_fodt, encoding="utf-8")

        # 7) FODT -> DOCX (better layout preservation than HTML->DOCX)
        soffice_fodt_to_docx(final_fodt_path, OUTPUT_DOCX)
        print(f"Done! Wrote {OUTPUT_DOCX.absolute()}")
    except Exception as e:
        print(f"Error: {e}")
        raise e
    finally:
        # Keep WORK_DIR if you want artifacts; else uncomment to clean:
        # shutil.rmtree(WORK_DIR, ignore_errors=True)
        pass

if __name__ == "__main__":
    main()
