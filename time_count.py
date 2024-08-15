import sys
import re

def calculate_times_and_lengths(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = file.read()
    except FileNotFoundError:
        print("File not found. Please check the path and try again.")
        return

    time_sums = {
        "cn tn": 0.0,
        "cn g2p": 0.0,
        "cn com": 0.0,
        "en tn": 0.0,
        "en g2p": 0.0,
        "en com": 0.0
    }
    text_lengths = []

    time_pattern = re.compile(r"(cn tn|cn g2p|cn com|en tn|en g2p|en com) time : ([\d\.e-]+)")
    length_pattern = re.compile(r"length of ori text :  (\d+)")

    matches = time_pattern.findall(data)
    for match in matches:
        time_sums[match[0]] += float(match[1])

    length_matches = length_pattern.findall(data)
    for length in length_matches:
        text_lengths.append(int(length))

    total_time = sum(time_sums.values())
    average_length = sum(text_lengths) / len(text_lengths) if text_lengths else 0

    # 计算每个时间段的平均时间
    num_texts = len(length_matches)
    average_times = {key: value / num_texts for key, value in time_sums.items()}

    print(f"Average text length: {average_length}")
    print(f"Total texts processed: {num_texts}")
    for key, value in time_sums.items():
        percentage = (value / total_time * 100) if total_time else 0
        average_time = average_times[key]
        print(f"Total time for {key}: {value:.6f} sec ({percentage:.2f}%)")
        print(f"Average time per text for {key}: {average_time:.6f} sec")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <file_path>")
    else:
        calculate_times_and_lengths(sys.argv[1])
