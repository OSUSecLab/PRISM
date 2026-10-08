from antlr4 import *
from antlr.CLexer import CLexer
from antlr.CParser import CParser
from antlr.CVisitor import CVisitor
import json
import sys
import re

arch = sys.argv[1]
opt = sys.argv[2]

var_predict = {}
lines = []

mark_line = -1

parameter_start_offset = []
parameter_end_offset = []



lastOffset = 0
param_list = []

def replace_variable_names(code, ori_variable_name, new_variable_name):
    try:
        variable_name_pattern = re.compile(r'([^a-zA-Z0-9_]|^)(%s)([^a-zA-Z0-9_])' % ori_variable_name)

        return variable_name_pattern.sub(r'\g<1>%s\g<3>'%new_variable_name, code)
    except re.error:
        return code

class ReplaceVisitor(CVisitor):
    def visitFunctionDefinition(self, ctx:CParser.FunctionDefinitionContext):
        global mark_line
        if ctx.declarator() is not None and mark_line == -1:
            if ctx.declarator().directDeclarator() is not None:
                if ctx.declarator().directDeclarator().directDeclarator() is not None:
                    functionNameCtx = ctx.declarator().directDeclarator().directDeclarator()
                    start_token = functionNameCtx.start
                    stop_token = functionNameCtx.stop
                    #print ('func', start_token.line, start_token.column, stop_token.line, stop_token.column)
                    lines[start_token.line - 1] = lines[start_token.line - 1][0: start_token.column] + '[MASK](' #+ \
                    #        lines[stop_token.line - 1][stop_token.column + len(stop_token.text):]
                    lastOffset = start_token.column
                    mark_line = stop_token.line - 1

                if ctx.declarator().directDeclarator().parameterTypeList() is not None:
                    if ctx.declarator().directDeclarator().parameterTypeList().parameterList() is not None:
                        if ctx.declarator().directDeclarator().parameterTypeList().parameterList().parameterDeclaration() is not None:
                            for temp in ctx.declarator().directDeclarator().parameterTypeList().parameterList().parameterDeclaration():
                                if temp.declarator() is not None and temp.declarator().directDeclarator().Identifier() is not None:
                                    varName = temp.declarator().directDeclarator().Identifier().getText()
                                    var_start = temp.declarator().directDeclarator().start
                                    var_stop = temp.declarator().directDeclarator().stop

                                    pointer_count = 0
                                    if temp.declarator().pointer() is not None:
                                        for star in temp.declarator().pointer().Star():
                                            pointer_count += 1

                                    for i in temp.declarationSpecifiers().declarationSpecifier():
                                        if i.typeSpecifier() is not None:
                                            varTypeCtx = i.typeSpecifier()
                                            start_token = varTypeCtx.start
                                            stop_token = varTypeCtx.stop
                                            varType = varTypeCtx.getText()
                                            if varName in var_predict:

                                                if len(var_predict[varName]) == 1:
                                                    continue
                                                if [var_predict[varName][1], var_predict[varName][0]] not in param_list:
                                                    param_list.append([var_predict[varName][1], var_predict[varName][0]])
                                            else:
                                                param_list.append([varType + '*'*pointer_count, varName])

        return self.visitChildren(ctx)

    def visitDeclaration(self, ctx:CParser.DeclarationContext):
        if ctx.declarationSpecifiers() is not None:
            if ctx.declarationSpecifiers().declarationSpecifier() is not None:
                for temp in ctx.declarationSpecifiers().declarationSpecifier():
                    if temp.typeSpecifier() is not None:
                        varTypeCtx = temp.typeSpecifier()
                        start_token = varTypeCtx.start
                        stop_token = varTypeCtx.stop
                        varType = varTypeCtx.getText()

                        if ctx.Identifier() is not None and ctx.Identifier().getText() in var_predict:
                            varName = ctx.Identifier().getText()
                            if len(var_predict[varName]) == 2:
                                var_stop_token = ctx.stop
                                lines[start_token.line - 1] = lines[start_token.line - 1][0: start_token.column] + f'{var_predict[varName][1]} {var_predict[varName][0]}' + \
                                        lines[stop_token.line - 1][var_stop_token.column + len(var_stop_token.text):]


                        if ctx.initDeclaratorList() is not None:
                            for i in ctx.initDeclaratorList().initDeclarator():
                                varName = i.declarator().getText().replace('*','')
                                if varName in var_predict and len(var_predict[varName]) == 2:
                                    var_start_token = i.declarator().start
                                    var_stop_token = i.declarator().stop
                                    if varName in var_predict:
                                        lines[start_token.line - 1] = lines[start_token.line - 1][0: start_token.column] + f'{var_predict[varName][1]} {var_predict[varName][0]}' + \
                                                lines[stop_token.line - 1][var_stop_token.column + len(var_stop_token.text) + lastOffset:]


        return self.visitChildren(ctx)



fp = open(f'var_output/{arch}_{opt}_var_infer.jsonl', 'r')
every_line = fp.readlines()
fp.close()

fp = open(f'var_output/{arch}_{opt}_var_replaced.jsonl', 'w')

cnt = 1
for line in every_line:
    print (cnt)
    cnt+= 1
    data = json.loads(line)

    var_predict = {}
    predicts = data["final"]
    for predict in predicts.split('\n'):
        if ":" not in predict or "WARNING" in predict:
            continue

        predict = predict.replace('</s>','').replace(':', '').replace(',', '')
        if len(predict.split()) <= 1:
            continue

        if len(predict.split()) == 2:
            var_cur = predict.split()[0]
            var_orig = predict.split()[1]
            var_predict[var_cur] = [var_orig]
            continue
        var_cur = predict.split()[0]
        var_orig = predict.split()[1]
        var_type = ' '.join(predict.split()[2:])
        var_predict[var_cur] = [var_orig, var_type]

    code = "\n".join(data["strip_decomp"].split('\n')[1:])
    if data["strip_decomp"] == "None":
        continue

    temp_fp = open(f'{arch}_{opt}', 'w')
    temp_fp.write(code)
    temp_fp.close()

    param_list = []
    temp_fp = open(f'{arch}_{opt}', 'r')
    new_data = temp_fp.read()
    antlrInput = InputStream(new_data)
    temp_fp.close()

    temp_fp = open(f'{arch}_{opt}', 'r')
    lines = temp_fp.readlines()
    temp_fp.close()

    try:
        lexer = CLexer(antlrInput)
        stream = CommonTokenStream(lexer)
        parser = CParser(stream)
        tree = parser.compilationUnit()
        visitor = ReplaceVisitor()
        visitor.visit(tree)
    except RecursionError:
        continue
    except IndexError:
        continue

    if len(param_list) == 0:
        #lines[0] = lines[0] + ')\n'

        lines[mark_line] = lines[mark_line] + ')\n'

    for idx in range(len(param_list)):
        lines[mark_line] = lines[mark_line] + f'{param_list[idx][0]} {param_list[idx][1]}'

        if idx == len(param_list) - 1:
            lines[mark_line] = lines[mark_line] + ')\n'
        else:
            lines[mark_line] = lines[mark_line] + ', '



    idx = mark_line + 1
    mark_line = -1

    while True:
        if '{' in lines[idx]:
            break
        lines.pop(idx)


    final_code = ''.join(lines)

    for key in var_predict.keys():
        name = var_predict[key][0]
        final_code = replace_variable_names(final_code, key, name)

    data["replaced"] = final_code
    json.dump(data, fp)
    fp.write('\n')
