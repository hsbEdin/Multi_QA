# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os

class Arguments:
    def __init__(self, confFile):
        if not os.path.exists(confFile):
            raise Exception("The argument file does not exist: " + confFile)
        self.confFile = confFile

    def is_int(self, s):
        try:
            int(s)
            return True
        except ValueError:
            return False

    def is_float(self, s):
        try:
            float(s)
            return True
        except ValueError:
            return False

    def is_bool(self, s):
        return s.lower() == 'true' or s.lower() == 'false'

    def readHyperDriveArguments(self, arguments):
        hyperdrive_opts = {}
        for i in range(0, len(arguments), 2):
            hp_name, hp_value = arguments[i:i+2]
            hp_name = hp_name.replace("--", "")
            if self.is_int(hp_value):
                hp_value = int(hp_value)
            elif self.is_float(hp_value):
                hp_value = float(hp_value)
            hyperdrive_opts[hp_name] = hp_value
        return hyperdrive_opts

    def readArguments(self):
        opt = {}
        with open(self.confFile, encoding='utf-8') as f:
            for line in f:
                l = line.replace('\t', ' ').strip() #用空格替换横向制表符，然后再去除每行首尾的空格
                if l.startswith("#"):  #判断屏蔽掉的代码
                    continue
                parts = l.split()  #以指定字符串进行切片，此处是空格（和\n）进行分割
                if len(parts) == 1:  #参数不为空，只有参数的key，没有值
                    key = parts[0]    #取参数的键
                    if not key in opt: #判断参数的键是否在opt中
                        opt[key] = True  #如果opt中没有当前键，将True赋值给该键
                if len(parts) == 2:  #有参数的key和值
                    key = parts[0]
                    value = parts[1]
                    # print(type(value)) value 为字符串格式
                    if not key in opt:
                        opt[key] = value
                        if self.is_int(value):  # 将字符串格式转化为 整型、浮点型和布尔值
                            opt[key] = int(value)
                        elif self.is_float(value):
                            opt[key] = float(value)
                        elif self.is_bool(value):
                            opt[key] = value.lower() == 'true'
                    else:
                        print('Warning: key %s already exists' % key)
        return opt


# # 调用类Arguments,产生对象 conf_args
# conf_args = Arguments('../conf')
#
# # 对对象 conf_args调用读参数的方法
# opt = conf_args.readArguments()