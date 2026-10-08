import os
import json
from ghidra.app.decompiler import DecompInterface, DecompileOptions
from ghidra.util.task import ConsoleTaskMonitor
from ghidra.program.model.pcode import FunctionPrototype
import sys

args = getScriptArgs()
comps = str(args[0]).split('/')

file_path = str(getProgramFile())
print("1. load the binary file: ", file_path)
#output_dir = os.path.join('strip_decomp', str(comps[-4]), str(comps[-3]), str(comps[-2]))
output_dir = 'strip_decomp'
assert (
    output_dir
), "Please provide the dir to save the results in 'decompilation/decom_for_stripped.py'"
output_file_path = os.path.join(output_dir, file_path.split('/')[-1] + '.json')
#output_file_path = 'strip_output.json'

if os.path.exists(output_file_path):
    print ('[-----------] file exists')
    sys.exit()

if not os.path.exists(output_dir):
    print("2. create the output folder: ", output_dir)
    os.makedirs(output_dir)

def get_data_type_info(f, var, is_arg, count):
    # variable name and type
    varname = var.getName()
    type_object = var.getDataType()
    type_name = type_object.getName()
    stack = "None"
    if var.isStackVariable():
        stack = var.getStackOffset()
    if var.isRegisterVariable():
        stack = var.getRegister().getName()
    #print (varname, stack)

    # get to what ever the pointer is pointing to
    ptr_bool = False
    for _ in range(type_name.count('*')):
        type_object = type_object.getDataType()
        type_name = type_object.getName()
        ptr_bool = True

    # if a typedef, get the primitive type definition
    try:
        type_object = type_object.getBaseDataType()
        type_name = type_object.getName()
    except:
        pass


    if ptr_bool:
        type_name += ' *'

    f[str(stack)] = {'type': str(type_name), 'name': str(varname)}


    return f


getCurrentProgram().setImageBase(toAddr(0), 0)
ref = currentProgram.getReferenceManager()
currentProgram = getCurrentProgram()
listing = currentProgram.getListing()
function = getFirstFunction()

ifc = DecompInterface()
options = DecompileOptions()
ifc.setOptions(options)
ifc.openProgram(currentProgram)

res = {}


print("3. decompile function: ")
while function is not None:
    #print('\t', function.name, function.getParameterCount(), function.getAutoParameterCount())
    funcname = function.name
    addrSet = function.getBody()
    codeUnits = listing.getCodeUnits(addrSet, True)

    all_vars = function.getAllVariables()
    all_args = function.getParameters()

    assembly = []

    for codeUnit in codeUnits:
        instruction = codeUnit.toString()
        assembly.append(instruction)

    # regular stack vars
    var_metadata = {}
    for var in all_vars:
        var_metadata = get_data_type_info(var_metadata, var, False, -1)


    decomp = ifc.decompileFunction(function, 60, ConsoleTaskMonitor())

    high = decomp.getHighFunction()
    if high is not None and high.getFunctionPrototype() is not None:
        for i in range(high.getFunctionPrototype().getNumParams()):
            param = high.getFunctionPrototype().getParam(i)
            param_name = param.getName()
            param_type = param.getDataType().getName()
            if param.getStorage() is not None and param.getStorage().getRegister() is not None:
                param_stack = param.getStorage().getRegister().getName()
                var_metadata[param_stack] = {'type': str(param_type), 'name': str(param_name)}

    if decomp.getDecompiledFunction() is not None:
        decompiled_function = decomp.getDecompiledFunction().getC()
    else:
        decompiled_function = "None"

    res[str(function.getEntryPoint())] = {
        #"assembly": assembly,
        "decomp_code": decompiled_function,
        "variable_metadata": var_metadata,
        #"args_metadata": args_metadata,
        'function_address': {
            'start': str(function.getEntryPoint()),
            'end': str(function.getBody().getMaxAddress()),
        },
        "func_name": funcname
    }

    function = getFunctionAfter(function)

with open(output_file_path, 'w') as f:
    print("4. write result to output_file_path: ", output_file_path)
    json.dump(res, f, indent=4)
