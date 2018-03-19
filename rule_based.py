import pandas as pd
import shutil
import re
import openpyxl
import os

company_file = ""
address_file = ""

class Item:
    addr = com = None
    cat1 = cat2 = None

    def __init__(self, key, parts):
        self.key = key
        self.parts = parts


def match_vendor_id(p):
    return re.match(r"(/[0-9A-Z])", p[:2])


def process_one_two(data):
    for key in data:
        item = data[key]
        pts = item.parts
        this_rule_applies = True
        for p in pts:
            if not (re.match(r"([1-3]/)|(/[0-9A-Z])", p[:2]) or len(p)==0):
                this_rule_applies = False
        if this_rule_applies:
            com, addr = [],[]
            for p in pts:
                prefix = p[:2]
                if prefix == "1/":
                    com.append( p[2:])
                elif prefix == "2/" or prefix == "3/":
                    addr.append(p[2:])
            item.cat1 = "1/2/3/ Tag"
            item.addr, item.com = " ".join(addr), " ".join(com)


def process_add(data):
    add_prefix = "ADD."
    for key in data:
        item = data[key]
        pts = item.parts
        if item.cat1 is None :
            add_part_idx = -1
            for i, p in enumerate(pts):
                if p[:4] == add_prefix:
                    add_part_idx = i
                    # print("ADD keyword", add_part_idx)
            if add_part_idx >=0:
                com, addr = [],[]
                for i, p in enumerate(pts):
                    if i < add_part_idx and not match_vendor_id(p):
                        com.append( p)
                    elif i == add_part_idx:
                        addr.append( p[4:])
                    elif i > add_part_idx:
                        addr.append( p)
                    # print("ADD keyword", add_part_idx, i,p,"@", com,"@", addr)
                item.cat1 = "ADD. keyword"
                item.addr, item.com = " ".join(addr), " ".join(com)


def process_com_end_sign(data):
    end_signs = ["INC", "LIMITED","LTD","CORP","LLC", "COMPANY", "LT", "PTE", "CO", "SA"]
    end_signs_r = [r'(\W|^)'+x+r'$' for x in end_signs]
    for key in data:
        item = data[key]
        pts = item.parts
        if item.cat1 is None:
            end_sign_part_idx = -1
            for i, p in enumerate(pts):
                p = re.sub(r'[(][^)]*[)]', '', p)
                p = re.sub(r'[^0-9A-Z ]', '', p).strip()
                for r in end_signs_r:
                    if re.search(r, p):
                        end_sign_part_idx = i
                        # print("Com keyword",end_sign_part_idx, re.search(r, p))
            if end_sign_part_idx >=0:
                com, addr = [],[]
                for i, p in enumerate(pts):
                    if i <= end_sign_part_idx and not match_vendor_id(p):
                        com.append( p)
                    elif i > end_sign_part_idx:
                        addr.append( p)
                    # print("Com keyword",end_sign_part_idx, i,p,"@", com,"@", addr)
                item.cat1 = "Company keyword"
                item.addr, item.com = " ".join(addr), " ".join(com)


def process_addr_start_sign(data):
    for key in data:
        item = data[key]
        pts = item.parts
        if item.cat1 is None:
            start_sign_part_idx = -1
            for i, p in enumerate(pts):
                if re.match(r'^\d[^/].*$', p):
                    start_sign_part_idx = i
            if start_sign_part_idx >=0:
                com, addr = [],[]
                for i, p in enumerate(pts):
                    if i < start_sign_part_idx and not match_vendor_id(p):
                        com.append( p)
                    elif i >= start_sign_part_idx:
                        addr.append( p)
                item.cat1 = "Leading number for address"
                item.addr, item.com = " ".join(addr), " ".join(com)


def process(df_, slug):
    data = {}
    keys = []
    for _, row in df_.iterrows():
        row_list = []
        for i in range(5):
            row_list.append(row[slug + str(i+1)].strip())
        key = re.sub(r"\s", "", "".join(row_list))
        keys.append(key)
        if key not in data:
            data[key] = Item(key, row_list)
    # df_["key"] = pd.Series(keys).values()
    # process data
    process_one_two(data)
    process_add(data)
    process_com_end_sign(data)
    process_addr_start_sign(data)

    # generate output
    cat1_list, addr_list, com_list = [], [], []
    for key in keys:
        item = data[key]
        cat1_list.append(item.cat1)
        addr_list.append(item.addr)
        com_list.append(item.com)
        if item.com and item.com.strip():
            print(item.com.strip(), file=open(company_file, 'a'))
        if item.addr and item.addr.strip():
            print(item.addr.strip(), file=open(address_file, 'a'))
    df_[slug + "company"] = pd.Series(com_list)
    df_[slug + "address"] = pd.Series(addr_list)
    df_[slug + "category"] = pd.Series(cat1_list)

def _write_excel(file_in, file_out, df, sheetname):
    # write output
    book = openpyxl.load_workbook(file_in)
    writer = pd.ExcelWriter(file_out, engine='openpyxl')
    writer.book = book
    writer.sheets = dict((ws.title, ws) for ws in book.worksheets)
    df.to_excel(writer, sheetname, index=False)
    writer.save()
def main():
    global company_file, address_file
    company_file = r"data\com.txt"
    address_file = r"data\add.txt"
    try:
        os.remove(company_file)
    except OSError:
        pass
    try:
        os.remove(address_file)
    except OSError:
        pass

    file_in = "C:\\mf\\geo_mapping\\PBOC RFI report\\Copy of MTS ordering and bene infor sample (2).xlsx"
    # file_in = "C:\\mf\\geo_mapping\\PBOC RFI report\\data_sm.xlsx"
    file_out = "C:\\mf\\geo_mapping\\PBOC RFI report\\output1.xlsx"
    sheetname = "IEMH"
    unique = False
    # shutil.copy2(file_in, file_out)
    # prepare dataframe
    df_ = pd.read_excel(file_in, sheetname=sheetname).fillna('')
    process(df_, "ORDERING_CUSTOMER_NAME_")
    process(df_, "BENEFICIARY_CUSTOMER_")
    # _write_excel(file_in, file_out, df_, sheetname)
if __name__ == '__main__':
    main()
    # data = {
    #     # "1": Item(None, ["/CUST/TH/MOC/0105555102703","BNB PROGRESS CO.,LTD.","ADD.99/385MOOBAN MANTHANA-ON NUT-","WONWAEN SOI SUKHAPIBAN 2 SOI 25","DOKM TH/BANGKOK THAILAND"]),
    #     # "2": Item(None, ["BAKER HUGHES (CHINA) OILFIELD  TECH","NOLOGY SERVICE CO., LTD","RMZ NXT CAMPUS,BLOCK 1-A,  3RD","FLOOR WHITEFIELD  BANGALORE INDIA",""]),
    #     # "3": Item(None, ["/002700019694","1/TYCO ELECTRONICS TECHNOLOGY (SIP)","1/ CO LTD","2/NO. 128 TINGLAN LANE, SUZHOU INDU","3/CN/SUZHOU JU 0000"]),
    #     # "4": Item(None, ["/730131331","CARNIVAL PLC","24303 TOWN CENTER DR","VALENCIA CA 91355 US",""]),
    #     # "5": Item(None, ["/10532414040042988","SUZHOU NAFCO PRECISION LIMITED","NO 269 SYNTHESIS PROTECTIVE TRADE","SECTION KUNSHAN CITY CHINA",""]),
    #     "6": Item(None, ["/0077620063","1/FORD MOTOR COMPANY OF SOUTHERN AF","2/SIMON VERMOOTEN ROAD","3/ZA/SILVERTON"]),
    #
    # }
    # process_one_two(data)
    # process_add(data)
    # process_com_end_sign(data)
    # process_addr_start_sign(data)