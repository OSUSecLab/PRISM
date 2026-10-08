import sys
import json
import os
import random
import argparse

from antlr4 import *
from antlr.CLexer import CLexer
from antlr.CParser import CParser
from antlr.CVisitor import CVisitor


lines = []
column_offset = {}

function_map = {}
function_count = 0

total = {}

class SubstituteFunctionNameVisitor(CVisitor):
    ### Substitute address related function name
    def visitPostfixExpression(self, ctx:CParser.PostfixExpressionContext):
        global lines
        global column_offset
        global function_count
        global function_map

        if ctx.LeftParen() is not None and ctx.primaryExpression() is not None:
            if ctx.primaryExpression().Identifier() is not None:
                functionNameToken = ctx.primaryExpression().Identifier().getSymbol()
                if functionNameToken.text.startswith('FUN_'):
                    if functionNameToken.text in function_map.keys():
                        new_function_name = function_map[functionNameToken.text]
                    else:
                        new_function_name = 'FUN_' + str(function_count)
                        function_map[functionNameToken.text] = new_function_name
                        function_count += 1
                    lines[functionNameToken.line - 1] = lines[functionNameToken.line - 1][0: functionNameToken.column + column_offset[functionNameToken.line - 1]] + new_function_name + lines[functionNameToken.line - 1][functionNameToken.column + len(functionNameToken.text) + column_offset[functionNameToken.line - 1]:]
                    column_offset[functionNameToken.line - 1] += len(new_function_name) - (len(functionNameToken.text))
        return self.visitChildren(ctx)


def substitute_decompiled(arch, opt, code):
    global lines
    global column_offset
    global function_map
    global function_count

    file = open(f'{arch}_{opt}_code.txt', 'w')
    file.write(code)
    file.close()

    file = open(f'{arch}_{opt}_code.txt', 'r')
    code = file.read()
    antlrInput = InputStream(code)
    file.close()

    file = open(f'{arch}_{opt}_code.txt', 'r')
    lines = file.readlines()
    file.close()

    for i in range(len(lines)):
        column_offset[i] = 0
    function_map = {}
    function_count = 0

    try:
        lexer = CLexer(antlrInput)
        stream = CommonTokenStream(lexer)
        parser = CParser(stream)
        tree = parser.compilationUnit()

        visitor = SubstituteFunctionNameVisitor()
        visitor.visit(tree)
    except RecursionError:
        return ""

    res = ""
    for line in lines:
        res += line

    os.remove(f'{arch}_{opt}_code.txt')

    return res


def main(args):
    error_fp = open('error', 'w')

    input_dir = os.path.join(args.input_dir, args.arch, args.opt)
    output_dir = os.path.join(args.output_dir, args.arch, args.opt)

    ### Divide Binary
    #train_part = 0.8
    #test_part = 0.1
    #validation_part = 0.1

    binary_names = []
    path_map = {}
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            binary_names.append(file)
            path_map[file] = os.path.join(root, file)


    ### use the saved division
    with open('division3.json', 'r') as f:
        data = json.load(f)
        train_binary = data['train_binary']
        test_binary = data['test_binary']
        validation_binary = data['valid_binary']


    print("[+] Training Set", train_binary)
    print("[+] Test Set", test_binary)
    print("[+] Validation Set", validation_binary)


    existed_function_name = []
    existed_function_body = []
    train = []
    test = []
    validation = []

    test_cnt = 0
    train_cnt = 0
    val_cnt = 0

    print("[+] Process Test Set Binary")
    for binary in test_binary:
        binary = binary + '.json'

        if binary not in path_map:
            error_fp.write(f'test {binary} not exists\n')
            continue

        with open(path_map[binary]) as f:
            data = json.load(f)

        binary_info = {}
        for function_name in data.keys():
            ### remove meaningless like 'FUN_00000f70'
            if 'FUN_' in function_name:
                continue

            ### delete duplicate name
            if function_name in existed_function_name:
                continue
            existed_function_name.append(function_name)

            decompiled_code = data[function_name]['stripped']
            if decompiled_code is None:
                error_fp.write(f'test {binary} stripped not exists\n')
                continue

            modified_decompiled_code = substitute_decompiled(args.arch, args.opt, decompiled_code)
            ### delete duplicate func content
            if modified_decompiled_code in existed_function_body:
                continue
            existed_function_body.append(modified_decompiled_code)

            sample = {}
            sample['variables'] = data[function_name]['variables']
            var_list = list(data[function_name]['variables'].keys())
            sample['strip_decomp'] = data[function_name]['strip_decomp']
            sample["resym_instruction"] = 'What are the original name and data type of variables '
            sample["resym_output"] = ""
            for idx in range(len(var_list)):
                var = var_list[idx]
                sample["resym_instruction"] += f"`{var}`"
                sample["resym_output"] += f"{var}: {data[function_name]['variables'][var]['name']}, {data[function_name]['variables'][var]['type']}"
                if idx != len(var_list)-1:
                    sample["resym_instruction"] += ", "
                    sample["resym_output"] += "\n"
                else:
                    sample["resym_instruction"] += "?\n```\n"




            sample["instruction"] = "Suppose you are an expert in software reverse engineering. Here is a piece of decompiled code, you should infer code semantics and tell me the original function name from the contents of the function to replace [MASK]. Now the decompiled codes are as follows:"
            sample["input"] = decompiled_code
            sample["output"] = function_name
            sample["unstripped"] = data[function_name]['unstripped']
            sample["stripped"] = data[function_name]['stripped']
            sample["binary"] = binary
            test.append(sample)
            test_cnt += 1
            binary_info[function_name] = sample
        total[binary] = binary_info


    print("[+] Process Valiation Set Binary")
    for binary in validation_binary:
        binary = binary + '.json'
        if binary not in path_map:
            error_fp.write(f'validate {binary} not exists\n')
            continue

        with open(path_map[binary]) as f:
            data = json.load(f)

        binary_info = {}
        for function_name in data.keys():
            ### remove meaningless like 'FUN_00000f70'
            if 'FUN_' in function_name:
                continue

            ### delete duplicate name
            if function_name in existed_function_name:
                continue
            existed_function_name.append(function_name)

            decompiled_code = data[function_name]['stripped']
            if decompiled_code is None:
                error_fp.write(f'validate {binary} stripped not exists\n')
                continue

            modified_decompiled_code = substitute_decompiled(args.arch, args.opt, decompiled_code)
            ### delete duplicate func content
            if modified_decompiled_code in existed_function_body:
                continue
            existed_function_body.append(modified_decompiled_code)

            sample = {}
            sample['variables'] = data[function_name]['variables']
            var_list = list(data[function_name]['variables'].keys())
            sample['strip_decomp'] = data[function_name]['strip_decomp']
            sample["resym_instruction"] = 'What are the original name and data type of variables '
            sample["resym_output"] = ""
            for idx in range(len(var_list)):
                var = var_list[idx]
                sample["resym_instruction"] += f"`{var}`"
                sample["resym_output"] += f"{var}: {data[function_name]['variables'][var]['name']}, {data[function_name]['variables'][var]['type']}"
                if idx != len(var_list)-1:
                    sample["resym_instruction"] += ", "
                    sample["resym_output"] += "\n"
                else:
                    sample["resym_instruction"] += "?\n```\n"



            sample["instruction"] = "Suppose you are an expert in software reverse engineering. Here is a piece of decompiled code, you should infer code semantics and tell me the original function name from the contents of the function to replace [MASK]. Now the decompiled codes are as follows:"
            sample["input"] = decompiled_code
            sample["output"] = function_name
            sample["unstripped"] = data[function_name]['unstripped']
            sample["stripped"] = data[function_name]['stripped']
            sample["binary"] = binary
            validation.append(sample)
            val_cnt += 1
            binary_info[function_name] = sample
        total[binary] = binary_info

    print("[+] Process Training Set Binary")
    for binary in train_binary:
        binary = binary + '.json'
        if binary not in path_map:
            error_fp.write(f'training {binary} not exists\n')
            continue

        with open(path_map[binary]) as f:
            data = json.load(f)

        binary_info = {}
        for function_name in data.keys():
            ### remove meaningless like 'FUN_00000f70' in training set
            if 'FUN_' in function_name:
                continue

            ### delete duplicate name
            if function_name in existed_function_name:
                continue
            existed_function_name.append(function_name)

            decompiled_code = data[function_name]['unstripped']
            if decompiled_code is None:
                error_fp.write(f'training {binary} unstripped not exists\n')
                continue

            modified_decompiled_code = substitute_decompiled(args.arch, args.opt, data[function_name]['stripped'])
            ### delete duplicate func content
            if modified_decompiled_code in existed_function_body:
                continue
            existed_function_body.append(modified_decompiled_code)

            sample = {}

            sample['variables'] = data[function_name]['variables']
            var_list = list(data[function_name]['variables'].keys())
            sample['strip_decomp'] = data[function_name]['strip_decomp']
            sample["resym_instruction"] = 'What are the original name and data type of variables '
            sample["resym_output"] = ""
            for idx in range(len(var_list)):
                var = var_list[idx]
                sample["resym_instruction"] += f"`{var}`"
                sample["resym_output"] += f"{var}: {data[function_name]['variables'][var]['name']}, {data[function_name]['variables'][var]['type']}"
                if idx != len(var_list)-1:
                    sample["resym_instruction"] += ", "
                    sample["resym_output"] += "\n"
                else:
                    sample["resym_instruction"] += "?\n```\n"


            sample["instruction"] = "Suppose you are an expert in software reverse engineering. Here is a piece of decompiled code, you should infer code semantics and tell me the original function name from the contents of the function to replace [MASK]. Now the decompiled codes are as follows:"
            sample["unstripped"] = data[function_name]['unstripped']
            sample["stripped"] = data[function_name]['stripped']
            sample["input"] = decompiled_code
            sample["output"] = function_name
            sample["binary"] = binary
            train.append(sample)
            train_cnt += 1
            binary_info[function_name] = sample
        total[binary] = binary_info



    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, 'training_set.json'), 'w') as f:

        for line in train:
            json.dump(line, f)
            f.write('\n')
        print("[+] Save training set to", os.path.join(output_dir, 'train_set.json'), train_cnt)
    with open(os.path.join(output_dir, 'test_set.json'), 'w') as f:
        for line in test:
            json.dump(line, f)
            f.write('\n')
        print("[+] Save test set to", os.path.join(output_dir, 'test_set.json'), test_cnt)
    with open(os.path.join(output_dir, 'validation_set.json'), 'w') as f:
        for line in validation:
            json.dump(line, f)
            f.write('\n')
        print("[+] Save validation set to", os.path.join(output_dir, 'validation_set.json'), val_cnt)

    with open(os.path.join(output_dir, 'complete.json'), 'w') as f:
        json.dump(total, f)
        print("[+] Save complete set")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Divide data into training, test and validation set.')
    parser.add_argument('-i', '--input_dir', type=str, required=True,
        # default='',
        help='Directory containing the combined decompiled code.')
    parser.add_argument('-o', '--output_dir', type=str, required=True,
        # default='',
        help='Directory to save the divided dataset.')
    parser.add_argument('-a', '--arch', type=str, default='x64')
    parser.add_argument('-opt', '--opt', type=str, default='O0')
    args = parser.parse_args()

    main(args)
