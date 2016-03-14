def removedup(filename):
    rcontent = open(filename, "r").readlines()
    header = rcontent[0]
    content = rcontent[1:]
    reduced_content = []
    for c in content:
        reduced_content.append(c.split(",")[:-1])

    dict_c = {}
    for c in content:
        key = ",".join(c.strip().split(",")[:-1])
        if key in dict_c.keys():
            print dict_c[key]
            print c.strip()
            print
        else:
            # print ">> ", key
            dict_c[key] = c.strip()

    content = header.strip()  + "\n"
    for key in dict_c.keys():
        content += dict_c[key]+ "\n"

    name = filename.split("/")[-1]
    mod_name = "./Data/"+name

    f = open(mod_name, "w")
    f.write(content)
    f.close()
#
# removedup("./Raw_Data/1_tp_read.csv")
# removedup("./Raw_Data/2_tp_write.csv")
# removedup("./Raw_Data/3_tp_read.csv")
# removedup("./Raw_Data/4_tp_write.csv")


from os import listdir
from os.path import isfile, join

dir = "./Raw_Data/"
files = [join(dir, f) for f in listdir(dir) if isfile(join(dir, f))]
for file in files: removedup(file)