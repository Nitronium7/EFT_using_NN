# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from datetime import datetime
import ROOT
import pandas as pd
import os
from tqdm import tqdm


def find_nodes(c_in, c_out):
    return 2 / 3 * c_in + c_out


def file_to_dataframe(file_name, extract_list, dataframe, sample_list, append_int):
    file_path = "~/data_download/" + file_name
    root_file = ROOT.TFile(file_path)
    tree = root_file.Get("1lep_preMerged")
    e_count = tree.GetEntries()
    c_count = len(extract_list)

    for i in tqdm(range(e_count)):
        tree.GetEntry(i)
        entry_list = []
        a_list = tree.sample
        sample_name = "".join(a_list)
        if sample_name in sample_list:
            entry_list.append(sample_name)
            for j in range(c_count - 2):
                entry_list.append((getattr(tree, extract_list[j + 1])))
            entry_list.append(append_int)
            dataframe.loc[len(dataframe)] = entry_list

    return dataframe


def splitter(X, Y):
    # randomly separates data into test and training data sets
    xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size=0.1, random_state=np.random.randint(0, 1000))
    return xTrain[:, [0, 1]], xTest[:, [0, 1]], xTrain[:, [2, 3, 4, 5]], xTest[:, [2, 3, 4, 5]], yTrain, yTest


def main():
    branches = ["sample", "lep_pt", "lep_eta", "lep_phi", "lep_m", "lep_q", "fatjet_pt",
                "fatjet_eta", "fatjet_phi", "fatjet_m", "nu_px", "nu_py", "nu_pz",
                "lep_helicity", "theta_r", "theta_lep", "phi_lep", "response"]

    order_list = ["SQ", ""]
    operator_list = ["cHq1", "cHS", "clq1", "cW"]

    for an_order in order_list:
        for an_operator in operator_list:
            EFT_string = "VVlvqq_NP" + an_order + "eq1_" + an_operator + "_1"
            sample_list_1 = [EFT_string]
            sample_list_2 = ["WW", "WZ"]
            tree_df = pd.DataFrame(columns=branches)
            tree_df = file_to_dataframe("1lep_EFT_x_r33-24.polarization_NN.root", branches, tree_df, sample_list_1, 1)
            tree_df = file_to_dataframe("1lep_diboson_x_r33-24.polarization_NN.root", branches, tree_df, sample_list_2,
                                        0)
            # print(tree_df)
            time_stamp = datetime.strftime(datetime.now(), "%m_%d_%Y_%H_%M_%S")
            file_name = an_order + "_" + an_operator + time_stamp + ".xlsx"
            out_path = "~/data_download/"
            out_file = os.path.join(out_path, file_name)
            tree_df.to_excel(out_file, index=False)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
