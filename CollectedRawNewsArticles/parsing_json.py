import ijson
import sys
import csv

def parseJson(input_path, output_path, no_of_files_to_generate):
    json_f = open(input_path)
    data_objects = ijson.items(json_f,"item")
    
    # date_inp = open(date_file, 'w')
    # writer = csv.writer(date_inp)
    count = 1

    #write content of each json object in a different file
    for data in data_objects:
        if "whole_content" not in data: continue
        description = data["whole_content"]
        brief = data["content"]
        if not description and not brief:
            continue
        data_to_write = ""
        if not description:
            data_to_write = brief
        else:
            data_to_write = description

        # file_name = data["source"] + "_" + data["_id"]["$oid"] + "_input.txt"

        if not data["title"] or not data_to_write or not data["publishAt"]["$date"]:
            continue
        # list_string  = data["title"][:100].split()
        # title = '_'.join(list_string)
        #file_name = title.replace('/','_') + "_" + data["publishAt"]["$date"][:10] + ".txt"
        fname = str(count) + ".txt"
        #if count>545530: output_path = "Test"
        f = open(output_path + "/" +fname,"w+")
        f.write(data_to_write)
        f.close()
        #writer.writerow(data["publishAt"]["$date"])
        
        
        if(int(no_of_files_to_generate)!=-1 and count%int(no_of_files_to_generate) == 0):
            print(str(count)+ " files generated..")
            break
        else:
            print(str(count)+ " files generated..")

        count += 1

    json_f.close()
    print("All files created successfully!!!!!!")

if __name__ == "__main__":
    # needs 2 argument from command line
    # 1.input_path contains json data
    # 2.output_path is folder that will contain all the ouput files 
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    no_of_files_to_generate = sys.argv[3]

    #date_file = sys.argv[4] if you wish to write the published date use this in a separate file

    parseJson(input_path,output_path, no_of_files_to_generate)
