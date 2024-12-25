import os
import random
from fastapi import APIRouter, HTTPException
from typing import List
from backend.data import router

# Define the base directories for testbeds
TESTBEDS_DIR = "./backend/data/mage/testbeds"
WILDER_DIR = "./backend/data/mage/wilder"

def find_testbeds(base_dir: str, is_wilder: bool = False):
    """
    Recursively find testbed files and organize them hierarchically.

    :param base_dir: Base directory to search for testbeds.
    :param is_wilder: Whether the testbeds are from the wilder dataset.
    :return: List of structured testbed metadata.
    """
    testbeds = []
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".csv"):  # Match CSV files
                relative_path = os.path.relpath(os.path.join(root, file), base_dir)
                path_parts = relative_path.split(os.sep)

                if is_wilder:
                    # Handle wilder testbeds (simple structure)
                    testbeds.append({
                        "testbed_name": path_parts[0],  # e.g., "T7_oog_set_gpt"
                        "file_name": file,
                        "file_path": relative_path,
                        "type": "wilder"
                    })
                else:
                    # Handle regular testbeds (multi-level structure)
                    testbeds.append({
                        "testbed_name": path_parts[0],  # e.g., "domain_specific"
                        "subtype": path_parts[1] if len(path_parts) > 2 else None,  # e.g., "model_A"
                        "file_name": file,
                        "file_path": relative_path,
                        "type": "ood" if "ood" in file else "regular"  # Determine type from file name
                    })
    # Sort testbeds by file path
    testbeds.sort(key=lambda x: x["file_path"])
    return testbeds

@router.get("/mage/testbeds")
def get_available_testbeds():
    """
    List all available testbeds in a structured format.
    """
    def organize_testbeds(testbeds):
        """
        Organize testbeds into a structured format with metadata.
        """
        organized = {}
        for testbed in testbeds:
            name = testbed["testbed_name"]
            if name not in organized:
                organized[name] = {
                    "testbed_name": name,
                    "subtypes": []
                }
            if testbed.get("subtype"):
                subtype = testbed["subtype"]
                organized[name]["subtypes"].append({
                    "subtype": subtype if testbed["type"] == "regular" else subtype + " (OOD)",
                    "file_name": testbed["file_name"],
                    "file_path": testbed["file_path"],
                    "type": testbed["type"]
                })
            else:
                organized[name]["subtypes"].append({
                    "subtype": "General",
                    "file_name": testbed["file_name"],
                    "file_path": testbed["file_path"],
                    "type": testbed["type"]
                })
        organized = list(organized.values())
        # Sort by testbed name
        organized.sort(key=lambda x: x["testbed_name"])
        return organized

    # Get testbeds and wilder testbeds
    testbeds = find_testbeds(TESTBEDS_DIR)
    # wilder_testbeds = find_testbeds(WILDER_DIR, is_wilder=True)

    # Organize testbeds and wilder testbeds
    return {
        "status": "success",
        "data": {
            "testbeds": organize_testbeds(testbeds)
        }
    }

def sample_data_from_testbed(testbed_path: str) -> str:
    """Sample a single data point from a testbed file."""
    try:
        with open(testbed_path, 'r') as file:
            lines = file.readlines()
            if not lines:
                raise ValueError(f"Testbed {testbed_path} is empty.")
            sampled_line = random.choice(lines)
            # The text may contain a comma, so we need to join the remaining parts
            text = ",".join(sampled_line.split(",")[:-1])
            label = sampled_line.split(",")[-1]
            return [text, label]

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Testbed not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@router.get("/mage/sample")
def sample_testbed_data(file_path: str):
    """
    Sample a single data point from a testbed file.

    - `file_path`: The full relative path to the testbed file.
    """
    label2decisions = {
        0: "machine-generated",
        1: "human-written",
    }
    
    # Validate the file path
    if not file_path:
        raise HTTPException(status_code=400, detail="File path is required.")

    # Determine the base directory (testbeds or wilder)
    base_dir = WILDER_DIR if file_path.startswith("wilder") else TESTBEDS_DIR

    # Construct the full file path
    testbed_path = os.path.join(base_dir, file_path)

    # Sample data
    try:
        text, label = sample_data_from_testbed(testbed_path)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading the file: {str(e)}")

    # Parse the sampled data (text, label)
    try:
        label = int(label)
    except ValueError:
        raise HTTPException(status_code=500, detail="Sampled data is not in the expected format: 'text,label'.")

    decision = label2decisions.get(label, "unknown")

    return {
        "file": file_path,
        "sampled_data": {
            "text": text.strip(),
            "label": label,
            "decision": decision,
        }
    }