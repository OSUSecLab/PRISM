import os.path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
import time
import pandas as pd
import glob
import subprocess
import argparse
import json
import sys
import os

thread_num = 15
executor = ThreadPoolExecutor(max_workers=thread_num)
ghidra_projects = [f'llm{i}/' for i in range(thread_num)]


def process_unstripped_binary(ghidra_path, project_path, project_name, binary_path):
    print(f"[*] hold {project_name} for {binary_path}")
    cmd = f"{ghidra_path} {project_path} {project_name} -import {binary_path} -readOnly -postScript ./decomp_for_unstripped.py {binary_path}"
    try:
        subprocess.run(cmd, shell=True, timeout=900*4)
    except subprocess.TimeoutExpired:
        print(f"[!] timeout for {binary_path}")
    proj = project_name.split('/')[0]
    os.system(f'rm ~/{proj}.lock*')
    ghidra_projects.append(project_name)
    print(f"[+] release {project_name} after finishing {binary_path}")


def process_stripped_binary(ghidra_path, project_path, project_name, binary_path):
    print(f"[*] hold {project_name} for {binary_path}")
    cmd = f"{ghidra_path} {project_path} {project_name} -import {binary_path} -readOnly -postScript ./decomp_for_stripped.py {binary_path}"
    try:
        subprocess.run(cmd, shell=True, timeout=900*4)
    except subprocess.TimeoutExpired:
        print(f"[!] timeout for {binary_path}")
    proj = project_name.split('/')[0]
    os.system(f'rm ~/{proj}.lock*')
    ghidra_projects.append(project_name)
    print(f"[+] release {project_name} after finishing {binary_path}")


def main(args):
    binary_path = args.binary_path
    if os.path.isfile(binary_path):
        print(f"[+] start to process {binary_path}")
        while len(ghidra_projects) == 0:
            print("Wait for ghidra project: 1 sec")
            time.sleep(1)
        ghidra_project = ghidra_projects.pop()
        executor.submit(process_unstripped_binary if args.unstripped else process_stripped_binary,
                        ghidra_path=args.ghidra_path,
                        project_path=args.project_path,
                        project_name=ghidra_project,
                        binary_path=binary_path,)
    elif os.path.isdir(binary_path):
        for root, dirs, files in os.walk(binary_path):
            for file in files:
                binary_file_path = os.path.join(root, file)

                if args.unstripped == True:
                    output_file = binary_file_path.replace('bin', 'unstrip_decomp', 1) + '.json'
                elif args.stripped == True:
                    output_file = binary_file_path.replace('strip', 'strip_decomp', 1) + '.json'

                if os.path.exists(output_file):
                    print(f"[+++] skip to process {binary_file_path}")
                    continue

                print(f"[+] start to process {binary_file_path}")
                while len(ghidra_projects) == 0:
                    print("Wait for ghidra project: 1 sec")
                    time.sleep(1)
                ghidra_project = ghidra_projects.pop()
                executor.submit(process_unstripped_binary if args.unstripped else process_stripped_binary,
                        ghidra_path=args.ghidra_path,
                        project_path=args.project_path,
                        project_name=ghidra_project,
                        binary_path=binary_file_path,)
                while executor._work_queue.qsize() > thread_num:
                    print("Wait for executor: 1 sec", executor._work_queue.qsize())
                    time.sleep(1)
    else:
        print(f"Check your {binary_path}.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform parallel decompilation and disassembling for binaries')
    parser.add_argument('-u', '--unstripped', action='store_true',
        help="Indicates that the binary is unstripped (contains debug symbols).")
    parser.add_argument('-s', '--stripped', action='store_true',
        help="Indicates that the binary is stripped (lacks debug symbols).")
    parser.add_argument('-b', '--binary_path', type=str, required=True,
        # default='',
        help="Specify the path to the binary file or folder containing binaries.")
    parser.add_argument('-g', '--ghidra_path', type=str, required=True,
        # default='',
        help="Provide the path to the Ghidra 'analyzeHeadless'.")
    parser.add_argument('-p', '--project_path', type=str, required=True,
        # default='',
        help="Specify the directory path to Ghidra projects.")
    args = parser.parse_args()

    if args.unstripped == True and args.stripped == True or args.unstripped == False and args.stripped == False:
        print("Error! You can just choose one mode '-u' or '-s'")
        sys.exit(0)

    if not os.path.exists(args.project_path):
        os.makedirs(args.project_path)

    main(args)
