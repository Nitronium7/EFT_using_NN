from bug import *


def run_net(train_df, test_df, epochs_i, epochs_f, model, GPU):
    # train = datasets.MNIST("", train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
    # test = datasets.MNIST("", train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))
    train = dataset_from_df(train_df, GPU)
    test = dataset_from_df(test_df, GPU)
    trainset = torch.utils.data.DataLoader(train, batch_size=32, shuffle=True)
    testset = torch.utils.data.DataLoader(test, batch_size=64, shuffle=False)
    accuracy_list = []
    loss_list = []
    print("Starting training...")
    print("*****************************************************************************")
    length, width = train_df.shape
    for epoch in (range(epochs_i, epochs_f)):
        torch.cuda.empty_cache()
        # optimizer = optim.Adam(model.parameters(), lr=(0.0001 * (1 / (4 + 0.2 * epoch))))
        #optimizer = optim.Adam(model.parameters(), lr=(0.01))

        optimizer = optim.Adam(model.parameters(), lr=(0.01 * (1 / (4 + 10 * epoch))))

        loss = 0
        loss_f = nn.BCEWithLogitsLoss()

        print("Epoch " + str(epoch + 1) + ":")
        # print(torch.cuda.memory_summary(device=None, abbreviated=False))
        model.train()
        for data in tqdm(trainset, leave=False):
            X, y = data
            model.zero_grad()
            output = model(X.view(-1, width - 1))
            y = y.unsqueeze(1)
            y = y.float()
            loss = loss_f(output, y)
            loss.backward()
            optimizer.step()

        print("Loss: ", round(loss.item(), 5), end="    ")
        loss_list.append(loss.item())

        with torch.no_grad():
            model.eval()
            correct = 0
            total = 0
            for data in testset:
                X, y = data
                output = model(X.view(-1, width - 1))
                for idx, i in enumerate(output):
                    if round(sigmoid(i.item())) == y[idx]:
                        correct += 1

                    total += 1
        accuracy = round(correct / total, 5)
        accuracy_list.append(accuracy)

        print("Accuracy:", accuracy)
        print("*****************************************************************************")
    return loss_list, accuracy_list, train_df, test_df


def test_method(data_arr, model, GPU):
    correct = 0
    total = 0
    correct_ = 0
    total_ = 0
    counter_ = 0
    hist_list_0 = []
    hist_list_1 = []
    counter = []
    counter2 = []
    accuracy_list = []

    length, width = data_arr.shape
    test = dataset_from_df(data_arr, GPU)
    complete_test_set = torch.utils.data.DataLoader(test, batch_size=32, shuffle=True)
    print("Testing...")

    model.eval()
    with torch.no_grad():
        for data in tqdm(complete_test_set):
            counter_ += 1
            X, y = data
            output = model(X.view(-1, width - 1))
            for idx, i in enumerate(output):
                value = sigmoid(i[0].item())

                if round(value) == y[idx]:
                    correct += 1
                    correct_ += 1

                if y[idx] == 0:
                    hist_list_0.append((i[0].item()))
                    counter.append(X)

                else:
                    hist_list_1.append((i[0].item()))
                    counter2.append(X)
                total += 1
                total_ += 1

            accuracy_ = round(correct_ / total_, 5)
            correct_ = 0
            total_ = 0
            accuracy_list.append(accuracy_)
    accuracy = round(correct / total, 5)
    print("Accuracy:", accuracy)
    print("*****************************************************************************")
    plot_hist(accuracy_list, 100)

    return hist_list_0, hist_list_1, counter, counter2


def export_data(hist_list_0, hist_list_1, counter, counter2, loss, accuracy, small_output):
    if small_output:
        data_1 = {'hist_list_0': hist_list_0, 'hist_list_1': hist_list_1}
    else:
        data_1 = {'hist_list_0': hist_list_0, 'hist_list_1': hist_list_1, 'c': counter, 'c2': counter2}
    data_2 = {'loss': loss, 'accuracy': accuracy}
    print("Exporting data...")
    export_df_1 = pd.DataFrame(data_1)

    export_df_2 = pd.DataFrame(data_2)
    time_stamp = datetime.strftime(datetime.now(), "%m_%d_%Y_%H_%M_%S")
    file_name_1 = "distribution" + time_stamp + ".csv"
    file_name_2 = "analysis" + time_stamp + ".csv"
    file_location = "output_files/"
    export_df_1.to_csv(file_location + file_name_1)
    export_df_2.to_csv(file_location + file_name_2)


def analyze_data(file_name, epoch_c_i, epoch_c_f, model, GPU):
    print(model)
    print("Parameters:", sum(param.numel() for param in model.parameters()))
    rb_tr, rb_te = data_processor(csv_to_data_frame(file_name, -1),
                                  "response", 0.9, False)
    loss, accuracy, train_set, test_set = run_net(rb_tr, rb_te, epoch_c_i, epoch_c_f, model, GPU)
    hist_list_0, hist_list_1, counter, counter2 = test_method(test_set, model, GPU)
    time_stamp = datetime.strftime(datetime.now(), "%m_%d_%Y_%H_%M_%S")
    net_name = "NN_model_" + time_stamp + ".pt"
    torch.save(model, net_name)
    plot_double_hist(hist_list_0, hist_list_1, 100, 0.25)
    plot_line(loss, "Loss over Epochs", "Epoch", "Loss")
    plot_line(accuracy, "Accuracy over Epochs", "Epoch", "Accuracy")
    export_data(hist_list_0, hist_list_1, counter, counter2, loss, accuracy, True)


def begin_analysis(file_name, epochs, GPU):
    print("Running begin_analysis...")
    enable_gpu = GPU
    if enable_gpu:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            device_message = "Running on " + torch.cuda.get_device_name()
            print(device_message)
        else:
            device = torch.device("cpu")
            print("Running on CPU")
    else:
        device = torch.device("cpu")
        print("Running on CPU")
    model = Net.Net().to(device)
    analyze_data(file_name, 0, epochs, model, GPU)


def continue_analysis(file_name, epoch_c_i, epoch_c_f, model_name, GPU):
    print("Running continue_analysis")
    enable_gpu = GPU
    if enable_gpu:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            device_message = "Running on " + torch.cuda.get_device_name()
            print(device_message)
        else:
            device = torch.device("cpu")
            print("Running on CPU")
    else:
        device = torch.device("cpu")
        print("Running on CPU")
    model = torch.load(model_name).to(device)

    analyze_data(file_name, epoch_c_i, epoch_c_f, model, GPU)


def test_analysis(model_name, file_name, GPU):
    print("Running test-analysis")
    test_model = torch.load(model_name)
    enable_gpu = GPU
    if enable_gpu:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            device_message = "Running on " + torch.cuda.get_device_name()
            print(device_message)
        else:
            device = torch.device("cpu")
            print("Running on CPU")
    else:
        device = torch.device("cpu")
        print("Running on CPU")
    test_model.to(device)
    test_model.eval()
    data_frame = csv_to_data_frame(file_name, -1)
    _, data_arr = data_processor(data_frame,
                                 "response", 0, False)

    test = dataset_from_df(data_arr, GPU)
    test_loader = torch.utils.data.DataLoader(test, batch_size=64, shuffle=True)

    hist_list_0, hist_list_1, counter, counter2 = test_method(data_arr, test_model, GPU)
    plot_double_hist(hist_list_0, hist_list_1, 100, 0.25)

    torch.set_grad_enabled(True)
    test_model.train()
    batch = next(iter(test_loader))
    test_data, _ = batch
    test_0 = test_data[:int(round(len(test_data) * 0.8))]
    test_1 = test_data[int(round(len(test_data) * 0.8)):len(test_data)]

    explainer = shap.DeepExplainer(test_model, test_0)
    shap_values = explainer.shap_values(test_1)
    feature_names = data_frame.columns.tolist()
    # plot the feature attributions
    feature_names.remove('sample')
    feature_names.remove('response')
    # print(shap_values)
    shap.summary_plot(shap_values, plot_type='bar', feature_names=feature_names)
    shap.summary_plot(shap_values, feature_names=feature_names)
    # shap.initjs()
    shap.force_plot(explainer.expected_value[0], shap_values[0], features=feature_names, matplotlib=True)
    # shap.plots.waterfall(shap_values)


def display_analysis(file_name):
    print("Running display_analysis")
    df = csv_to_data_frame(file_name, -1)
    hist_list_0, hist_list_1 = df['hist_list_0'].tolist(), df['hist_list_1'].tolist()
    plot_double_hist(hist_list_0, hist_list_1, 100, 0.25)
    count = 0
    for each in hist_list_0:
        if each < 0:
            count += 1
    for each in hist_list_1:
        if each >= 0:
            count += 1

    accuracy = round(count / (len(hist_list_0) * 2), 5)
    print("Accuracy:", accuracy)


def display_sweep(model_name, file_name, GPU):
    print("Running display_sweep")
    df = csv_to_data_frame(file_name, -1)
    array = test_processor(df)
    _, width = array.shape
    test_model = torch.load(model_name)
    enable_gpu = GPU
    if enable_gpu:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            device_message = "Running on " + torch.cuda.get_device_name()
            print(device_message)
        else:
            device = torch.device("cpu")
            print("Running on CPU")
    else:
        device = torch.device("cpu")
        print("Running on CPU")
    test_model.to(device)
    test_model.eval()
    nn_score_list = []

    with torch.no_grad():

        for i in tqdm(range(len(df))):
            test_case = dataset_from_df(np.expand_dims(array[i], axis=0), GPU)
            case_loader = torch.utils.data.DataLoader(test_case, batch_size=1, shuffle=True)
            for data in (case_loader):
                X, y = data
                output = test_model(X.view(-1, width - 1))
                for _, m in enumerate(output):
                    nn_score_list.append(sigmoid(m[0].item()))
    df["nn_score"] = nn_score_list
    time_stamp = datetime.strftime(datetime.now(), "%m_%d_%Y_%H_%M_%S")
    file_name = "output_files/" + "output_nn_scores" + time_stamp + ".csv"
    df.to_csv(file_name)


def graphs_analysis(file_name, cutoff):
    df = csv_to_data_frame(file_name, -1)
    tp = df.loc[(df['response'] == 1) & (df['nn_score'] > cutoff)]
    tn = df.loc[(df['response'] == 0) & (df['nn_score'] < cutoff)]
    fp = df.loc[(df['response'] == 0) & (df['nn_score'] > cutoff)]
    fn = df.loc[(df['response'] == 1) & (df['nn_score'] < cutoff)]
    bins = 20
    for each in df.columns:
        plt.hist(tp.loc[:, each], bins, label='tp', histtype=u'step', density=True, color='b')
        plt.hist(tn.loc[:, each], bins, label='tn', histtype=u'step', density=True, color='pink')
        plt.hist(fp.loc[:, each], bins, label='fp', histtype=u'step', density=True, color='y')
        plt.hist(fn.loc[:, each], bins, label='fn', histtype=u'step', density=True, color='r')
        plt.legend(loc='upper right')
        plt.title(each)
        plt.show()
    return


def choose_program(count):
    GPU = False
    if count == 1:
        begin_analysis("SQ_cW_v1_train.csv", 10, GPU)
    if count == 2:
        continue_analysis("btc_test_Jul22.csv", 50, 50, "stock_model_1_40e.pt", GPU)
    if count == 3:
        test_analysis("NN_model_10_23_2022_03_58_33.pt", "SQ_cW_v1_test.csv", GPU)
    if count == 4:
        display_analysis("output_files/distribution08_08_2022_14_30_31.csv") #deprecated
    if count == 5:
        display_sweep("NN_model_10_23_2022_03_05_56.pt", "SQ_cW_v1.csv", GPU)
    if count == 6:
        graphs_analysis("output_files/output_nn_scores10_23_2022_03_11_57.csv", 0.6)
    return

def main():
    choose_program(3)
    input("Press Enter to exit...")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
