import os
from natsort import natsorted
import json
import re
from collections import defaultdict
import multiprocessing
import json
from collections import defaultdict
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from easydict import EasyDict as edict

from hrse.mllm.gemini_api import GeminiSingle, GeminiPerspective, GeminiMulti
from hrse.mllm.qwen_api import QwenSingle, QwenPerspective, QwenMulti, Qwen
from hrse.mllm.openai_compat_api import QnAIGCVision
from hrse.mllm.prompts import *

def load_gt_contacts(json_path):
    with open(json_path, 'r') as f:
        gt = json.load(f)
    gt_dict = {}
    for item in gt["contacts"]:
        gt_dict[str(item["frame"]).zfill(5)] = {
            "r_contact": item["r_contact"],
            "l_contact": item["l_contact"]
        }
    return gt_dict


def parse_all_results(qwen_results, sequential_k=None):
    frame_contacts = defaultdict(list)
    # Regex that supports optional fingers fields.
    contact_pattern = re.compile(
        r'"frame"\s*:\s*"?(\d+)"?,\s*"r_contact"\s*:\s*(true|false),\s*"l_contact"\s*:\s*(true|false)(?:,\s*"r_fingers"\s*:\s*\[([^\]]*)\])?(?:,\s*"l_fingers"\s*:\s*\[([^\]]*)\])?'
    )
    for img_name, result_str in qwen_results.items():
        matches = list(contact_pattern.finditer(result_str))
        expected_frames = None
        sample_match = re.search(r"sampleRGBD_(\d+)", img_name)
        if sequential_k and sample_match and len(matches) == sequential_k:
            start_frame = int(sample_match.group(1)) - 1
            expected_frames = list(range(start_frame, start_frame + sequential_k))
        for match_idx, m in enumerate(matches):
            frame = (
                expected_frames[match_idx]
                if expected_frames is not None
                else int(m.group(1))
            )
            r_contact = m.group(2) == "true"
            l_contact = m.group(3) == "true"
            r_fingers = re.findall(r'"(thumb|index|middle)"', m.group(4) or "")
            l_fingers = re.findall(r'"(thumb|index|middle)"', m.group(5) or "")
            frame_contacts[frame].append({
                "r_contact": r_contact,
                "l_contact": l_contact,
                "r_fingers": r_fingers,
                "l_fingers": l_fingers
            })

    # Merge results for the same frame.
    # For finger contacts, choose the most frequent combination.
    # On ties, choose the combination with fewer fingers; if still tied, choose the latest one.
    contacts = []
    for frame in sorted(frame_contacts.keys()):
        items = frame_contacts[frame]
        r_true = sum(item["r_contact"] for item in items)
        r_false = len(items) - r_true
        l_true = sum(item["l_contact"] for item in items)
        l_false = len(items) - l_true
        # Majority vote: True if positives are the strict majority; otherwise False.
        r_contact = True if r_true > r_false else False
        l_contact = True if l_true > l_false else False

        # Count occurrences of every r_fingers combination.
        r_finger_counts = defaultdict(int)
        r_finger_last_idx = {}
        for idx, item in enumerate(items):
            key = tuple(sorted(item["r_fingers"]))
            r_finger_counts[key] += 1
            r_finger_last_idx[key] = idx
        # Find the most frequent combination.
        max_count = max(r_finger_counts.values()) if r_finger_counts else 0
        candidates = [k for k, v in r_finger_counts.items() if v == max_count]
        # If multiple candidates remain, choose the one with fewer fingers.
        min_len = min(len(k) for k in candidates) if candidates else 0
        candidates2 = [k for k in candidates if len(k) == min_len]
        # If still tied, choose the latest one.
        if candidates2:
            chosen_r_fingers = max(candidates2, key=lambda k: r_finger_last_idx[k])
        else:
            chosen_r_fingers = ()

        # Apply the same rule to l_fingers.
        l_finger_counts = defaultdict(int)
        l_finger_last_idx = {}
        for idx, item in enumerate(items):
            key = tuple(sorted(item["l_fingers"]))
            l_finger_counts[key] += 1
            l_finger_last_idx[key] = idx
        max_count = max(l_finger_counts.values()) if l_finger_counts else 0
        candidates = [k for k, v in l_finger_counts.items() if v == max_count]
        min_len = min(len(k) for k in candidates) if candidates else 0
        candidates2 = [k for k in candidates if len(k) == min_len]
        if candidates2:
            chosen_l_fingers = max(candidates2, key=lambda k: l_finger_last_idx[k])
        else:
            chosen_l_fingers = ()

        contacts.append({
            "frame": frame,
            "r_contact": r_contact,
            "l_contact": l_contact,
            "r_fingers": list(chosen_r_fingers),
            "l_fingers": list(chosen_l_fingers)
        })

    # After majority voting: decide appeared based on final contacts
    appeared_set = set()
    if any(c.get("l_contact", False) for c in contacts):
        appeared_set.add("left")
    if any(c.get("r_contact", False) for c in contacts):
        appeared_set.add("right")

    return {
        "frames_cnt": len(contacts),
        "appeared": sorted(list(appeared_set)),
        "contacts": contacts
    }

def get_contact_segments(contacts, hand):
    segments = []
    in_contact = False
    start = None
    fingers_set = set()
    for c in contacts:
        contact = c[f"{hand}_contact"]
        fingers = set(c.get(f"{hand}_fingers", []))
        if contact:
            if not in_contact:
                in_contact = True
                start = c["frame"]
                fingers_set = set(fingers)
            else:
                fingers_set |= fingers
        else:
            if in_contact:
                segments.append({"start": start, "end": c["frame"]-1, "fingers": sorted(list(fingers_set))})
                in_contact = False
                start = None
                fingers_set = set()
    if in_contact:
        segments.append({"start": start, "end": contacts[-1]["frame"], "fingers": sorted(list(fingers_set))})
    return segments


def fill_missing_contact_fingers(contacts, hand):
    """Fill an uncertain contact frame from the nearest labeled contact frame."""
    contact_key = f"{hand}_contact"
    fingers_key = f"{hand}_fingers"
    for index, contact in enumerate(contacts):
        if not contact.get(contact_key) or contact.get(fingers_key):
            continue

        candidates = []
        for direction in (-1, 1):
            cursor = index + direction
            distance = 1
            while 0 <= cursor < len(contacts):
                neighbor = contacts[cursor]
                if not neighbor.get(contact_key):
                    break
                fingers = neighbor.get(fingers_key, [])
                if fingers:
                    candidates.append((distance, len(fingers), cursor, fingers))
                    break
                cursor += direction
                distance += 1

        if candidates:
            # Prefer the nearest observation; ties follow the conservative
            # parser rule and choose fewer active fingers.
            _, _, _, fingers = min(candidates)
            contact[fingers_key] = list(fingers)


def save_vlm_result_to_json(vlm_result, out_path):
    with open(out_path, "w") as f:
        json.dump(vlm_result, f, indent=2, ensure_ascii=False)
        

def count_files(directory):
        try:
            return sum(
                1 for entry in os.listdir(directory)
                if os.path.isfile(os.path.join(directory, entry))  # Ensure this is a file.
                and "RGBD" in entry
                and entry.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))  # Match image formats.
            )
        except FileNotFoundError:
            print(f"Directory '{directory}' does not exist")
            return 0


def process_dataset(args):
    seq_name = args.seq_name
    qwen_results = {}
    use_qnaigc = bool(os.getenv("QNAIGC_API_KEY"))
    if use_qnaigc:
        perspective_client = QnAIGCVision()
        contact_client = perspective_client
        print(
            "Using QnAIGC OpenAI-compatible backend with model "
            f"{os.environ.get('QNAIGC_MODEL')}"
        )
    else:
        perspective_client = QwenPerspective()
        contact_client = QwenSingle()
    if os.path.exists(args.mllm_path):
        num_files = len([f for f in os.listdir(args.mllm_path) if os.path.isfile(os.path.join(args.mllm_path, f))])
        print(f"[{seq_name}] RGBD image count: {int(num_files / 2)}")
    else:
        print(f"[{seq_name}] Folder {args.mllm_path} does not exist")

    image_paths = [
        os.path.join(args.mllm_path, fname)
        for fname in natsorted(os.listdir(args.mllm_path))
        if "RGBD" in fname and fname.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))
    ][:10]
    perspective = args.perspective
    if perspective == "auto":
        print("=== Perspective inference ===")
        qwen_response_p = perspective_client.request_with_images(
            prompt_perspective, image_paths
        )
        print("Perspective response:", qwen_response_p, " for seq ", seq_name)
        perspective = str(qwen_response_p).strip()
        if perspective not in ["1", "3"]:
            print("Warning: perspective output not recognized, defaulting to 1st person.")
            perspective = "1"
    print("Using perspective:", perspective)

    if perspective == "3":
        hand_hint = Third_Person_Perspective_Hand_Hint
    else:
        # print("Using 1st person hand hint for ", seq_name)
        hand_hint = First_Perspective_Hand_Hint

    prompt_with_perspective = f"""
        this is a horizontally merged sequence of K selected frame from a video which id from a {'third' if perspective == '3' else 'first'}-person perspective.
    {hand_hint}
    {prompt}
    """

    response_dir = args.response_dir or os.path.join(
        args.out, "mllm_raw", seq_name
    )
    os.makedirs(response_dir, exist_ok=True)
    contact_images = [
        os.path.join(args.mllm_path, f"sampleRGBD_{i}.jpg")
        for i in range(1, count_files(args.mllm_path) + 1)
    ]

    def infer_contact(img_path):
        image_name = os.path.basename(img_path)
        response_path = os.path.join(
            response_dir, os.path.splitext(image_name)[0] + ".txt"
        )
        if os.path.isfile(response_path) and os.path.getsize(response_path) > 0:
            with open(response_path, "r") as f:
                return image_name, f.read(), True
        response = contact_client.request_with_image(
            prompt_with_perspective, img_path
        )
        temp_path = response_path + ".tmp"
        with open(temp_path, "w") as f:
            f.write(response)
        os.replace(temp_path, response_path)
        return image_name, response, False

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(infer_contact, img_path): img_path
            for img_path in contact_images
        }
        completed = 0
        for future in as_completed(futures):
            image_name, response, cached = future.result()
            qwen_results[image_name] = response
            completed += 1
            print(
                f"Contact response {completed}/{len(contact_images)}: "
                f"{image_name} ({'cached' if cached else 'API'})"
            )
    # return 

    vlm_result = parse_all_results(
        qwen_results, sequential_k=args.sequential_k
    )
    fill_missing_contact_fingers(vlm_result["contacts"], "l")
    fill_missing_contact_fingers(vlm_result["contacts"], "r")
    vlm_result["contact_segments"] = {
        "left": get_contact_segments(vlm_result["contacts"], "l"),
        "right": get_contact_segments(vlm_result["contacts"], "r")
    }
    appeared = set(vlm_result.get("appeared", []))
    for contact in vlm_result["contacts"]:
        if "left" not in appeared:
            contact["l_fingers"] = []
        if "right" not in appeared:
            contact["r_fingers"] = []
    for seg in vlm_result["contact_segments"].get("left", []):
        if "left" not in appeared:
            seg["fingers"] = []
    for seg in vlm_result["contact_segments"].get("right", []):
        if "right" not in appeared:
            seg["fingers"] = []
    os.makedirs(args.out, exist_ok=True)
    save_vlm_result_to_json(vlm_result, f"{args.out}/ho_contact.json")
    
    # uncomment below to run evaluation
    """
    gt_result = load_gt_contacts(f"/hrse-seq_name/{seq_name}/processed/ho_contact.json")
    acc, wrong_frames = calc_vlm_accuracy_multi(vlm_result, gt_result)
    fp_rate_left, fp_rate_right, fp_rate_any, fp_frames = calc_vlm_fp_rate_multi(vlm_result, gt_result)

    print(f"[{seq_name}] Accuracy: {acc:.4f}")
    print(f"[{seq_name}] FP rate (any hand): {fp_rate_any:.4f}")
    return (seq_name, acc, fp_rate_any)
    """
    return (seq_name, 0, 0) 

def main():
    parser = argparse.ArgumentParser(description="Set experiment name and paths")
    parser.add_argument("--mllm-path", type=str, required=True, help="Path to the input sequence")
    parser.add_argument("--seq-name", type=str, default="default_seq_name", help="Name of the sequence (for logging and output naming)")
    parser.add_argument("--out", type=str, default='outputs/mllm_contact')
    parser.add_argument(
        "--response-dir", type=str, default=None,
        help="Directory for resumable raw model responses",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Number of concurrent API requests",
    )
    parser.add_argument(
        "--perspective", choices=["auto", "1", "3"], default="auto",
        help="Camera perspective override: 1=first-person, 3=third-person",
    )
    parser.add_argument(
        "--sequential-k", type=int, default=None,
        help="Correct frame ids using sequential K-frame window names",
    )
    args = parser.parse_args()
    args = edict(vars(args))
    # cpu_count = max(int(multiprocessing.cpu_count() / 2), 1)
    # with multiprocessing.Pool(cpu_count) as pool:
    #     acc_table = pool.map(process_dataset, args.seq_path)
    acc_table = [process_dataset(args)] 
    print("\n==== Accuracy Table ====")
    print("Dataset\tAccuracy\tFP_anyhand")
    for seq_name, acc, fp_rate_any in acc_table:
        print(f"{seq_name}\t{acc:.4f}\t{fp_rate_any:.4f}")

if __name__ == '__main__':
    main()
