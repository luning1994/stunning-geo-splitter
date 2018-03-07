import pandas as pd
import shutil
import re

def process_one_two(data):
    for key in data:
        parts = data[key]["parts"]
        this_rule_applies = True
        for part in parts:
            if not re.match(r"([0-9]/)|(/[0-9A-Z])", part[:2]):
                this_rule_applies = False
        if this_rule_applies:
            com = addr = ""
            for part in parts:
                if part[:2] == "1/":
                    com += part[2:]
                if part[:2] == "2/":
                    addr += part[2:]
            data[key]["cat1"] = "12"
            data["addr_str"] = 
    pass


def process_add(data):
    pass


def process_com_end_sign(data):
    pass


def process_addr_start_sign(data):
    pass


if __name__ == '__main__':
    data = {}
    file_in = "C:\\mf\\geo_mapping\\PBOC RFI report\\Copy of MTS ordering and bene infor sample (2).xls"
    # shutil.copy2(file_in, file_out)
    df_ = pd.read_excel(file_in).fillna('')
    for _, row in df_.iterrows():
        row_list = []
        for i in range(5):
            row_list.append(row["ORDERING_CUSTOMER_NAME_" + str(i+1)].strip())
        key = ""
        for part in row_list:
            key += part
        key = re.sub(r"\s", "", key)
        if key not in data:
            data[key] = {
                "parts": row_list,
                "addr_str":None,
                "com_str": None,
                "cat1": None,
                "cat2": None
            }
    process_one_two(data)
    process_add(data)
    process_com_end_sign(data)
    process_addr_start_sign(data)