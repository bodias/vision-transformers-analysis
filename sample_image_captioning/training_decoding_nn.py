import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

#Import svm model
from sklearn import svm
from sklearn import metrics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
from itertools import product

EXP_FOLDER = "experiments"

def base_model(model="NN", n_classes = 5):
    if model == "NN":
        clf_model = Sequential()
        clf_model.add(Input(shape=(768)))
        clf_model.add(Dense(256, activation='relu'))
        clf_model.add(Dropout(0.5))
        clf_model.add(Dense(n_classes))
        clf_model.add(Activation("softmax"))
        sgd = SGD(learning_rate=0.0001, momentum=0.9, nesterov=True)
        # adam = Adam()
        clf_model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])        
    else:
        clf_model = svm.SVC(kernel='linear') # Linear Kernel
    return clf_model

def prepare_dataset(dataset_folder, filter_caption=None):
    # files with features and labels are split due to ram limitations
    # on generation it has to fit ram, and also on reading
    files = os.listdir(dataset_folder)
    features = pd.DataFrame()
    for file in files:
        ## WORKAROUND 'feat-tokens_act-1-10.pickle'
        if int(file[16])==1:
            print(f"person dataset {file[16]}")
            first_label = "person"
        else:
            print(f"car dataset {file[16]}")
            first_label = "car"

        print(f"Processing file '{file}'")    
        obj_features = pd.read_pickle(os.path.join(dataset_folder, file))
        obj_features["class"] = obj_features["class"].apply(lambda x: first_label+"-"+x)
        if filter_caption is not None:
            print(f"filter caption is on:{filter_caption}")
            obj_features = obj_features[obj_features["caption_filter"]==filter_caption]
        else:
            # order by caption filter to make sure there's caption_filter since only a few have
            obj_features = obj_features.sort_values(by=['caption_filter'], ascending=False)
            obj_features = obj_features.reset_index(drop=True)
        obj_features = obj_features[:1000]
        features = pd.concat([features, obj_features])

    features = features.reset_index(drop=True)
    # TODO: fix consistent token selection with multiple layers
    features = features[(~features["second_fg_tokens"].isnull()) & 
                        (~features["main_fg_tokens"].isnull())
    #                     (~features["second_consistent_fg_token"].isnull()) &
    #                     (~features["main_consistent_fg_token"].isnull())
                        ]
    
    labels = features['class'].values.tolist()
    unique_labels = sorted(list(set(labels)))
    labels_to_idx = dict(zip(unique_labels, range(len(unique_labels))))

    features["labels"] = features['class'].apply(lambda x: labels_to_idx[x])
    # for stratification based on labels + caption
    features["labels_caption"] = features["class"].astype(str) + features["caption_filter"].astype(str)
    features = features.reset_index(drop=True)

    return features, unique_labels

def dataset_split(features, testset_file=""):
    if testset_file:
        print(f"Loading testset from {testset_file}")
        with open(testset_file, 'rb') as handle:
            testset = pickle.load(handle) 
        train_idx, train_labels = testset["train"]["data"], testset["train"]["labels"]
        test_idx, test_labels = testset["test"]["data"], testset["test"]["labels"]
    else: 
        train_idx, test_idx, train_labels, test_labels = train_test_split(features.index.tolist(), 
                                                                          features["labels"].tolist(), 
                                                                          test_size=0.10, 
                                                                          stratify=features["labels_caption"].tolist(),
                                                                          random_state=42, 
                                                                          shuffle=True)
    # validation data
    train_idx, val_idx, train_labels, val_labels = train_test_split(features.filter(items=train_idx, axis=0).index.tolist(), 
                                                                    features.filter(items=train_idx, axis=0)["labels"].tolist(), 
                                                                    test_size=0.10, 
                                                                    stratify=features.filter(items=train_idx, axis=0)["labels_caption"].tolist(),
                                                                    random_state=42, 
                                                                    shuffle=True)   

    return train_idx, val_idx, test_idx, train_labels, val_labels, test_labels

def plot_training_curves(histories, filename):
    fig, axs = plt.subplots(nrows=len(histories), ncols=2, figsize=(8, 4*len(histories)))
    for idx, (layer_name, hist) in enumerate(histories.items()):
        axs[idx, 0].plot(hist.history['loss'])
        axs[idx, 0].plot(hist.history['val_loss'])
        axs[idx, 0].set_title(f'{layer_name} loss')
    #     axs[idx, 0].ylabel('loss')
    #     axs[idx, 0].xlabel('epoch')
        axs[idx, 0].legend(['train', 'val'], loc='upper left')
        axs[idx, 1].plot(hist.history['accuracy'])
        axs[idx, 1].plot(hist.history['val_accuracy'])
        axs[idx, 1].set_title(f'{layer_name} accuracy')
    #     axs[idx, 1].ylabel('accuracy')
    #     axs[idx, 1].xlabel('epoch')
        axs[idx, 1].legend(['train', 'val'], loc='upper left')

    plt.savefig(fname=filename)
    # plt.show()
    plt.close('all')

def save_test_scores(scores, exp_name, filename, class_labels = None):
    scores_pd = {'model_name': [], 'model': [], 'object': [], 'token_strategy': [], 'hidden_state_layer': []}
    if class_labels is None:
        scores_pd['test_score'] = []
        scores_pd['loss'] = []
    else:
        for label in class_labels:
            scores_pd[label] = []

    for p, score in scores.items():
        scores_pd['model_name'].append(exp_name)
        strategy = p[0]
        layer = p[1]
        obj = p[2]
        model = p[3]
        scores_pd['hidden_state_layer'].append(layer)
        scores_pd['object'].append(obj)
        scores_pd['token_strategy'].append(strategy)
        scores_pd['model'].append(model)
        if class_labels is None:
            scores_pd['loss'].append(score[0])
            scores_pd['test_score'].append(score[1])
        else:
            for cls_idx, label in enumerate(class_labels):
                scores_pd[label].append(score[cls_idx])

    scores_pd = pd.DataFrame(scores_pd)
    scores_pd.to_csv(filename, index=False)
    return scores_pd    

def run_experiments(features,
                    labels,
                    exp_name,
                    unique_label_names = None,
                    token_strategies = ['max_image', 'max_obj', 'min_obj', 'random_obj'],
                    layers = [3, 4, 9, 10, 11],
                    objects = ['main', 'second'],
                    models = ['NN'],
                    epochs = 60):
    
    print(f"dataset size: {len(features)}")
    train_idx, val_idx, test_idx, train_labels, val_labels, test_labels = dataset_split(features)
    
    # create a cartesian product with all parameters
    params = list(product(token_strategies, layers, objects, models))

    histories = {}
    test_scores = {}
    class_scores = {}
    
    for exp_n, p in enumerate(params):
        strategy = p[0]
        layer = p[1]
        obj = p[2]
        model = p[3]

        if model=="NN":
            train_y = to_categorical(train_labels)
            val_y = to_categorical(val_labels)
            test_y = to_categorical(test_labels)
        else:
            train_y = train_labels
            val_y = val_labels
            test_y = test_labels

        train_x = features.filter(items=train_idx, axis=0)[f"{obj}_fg_tokens_act"].apply(lambda x: x[layer][strategy]).to_numpy()
        val_x = features.filter(items=val_idx, axis=0)[f"{obj}_fg_tokens_act"].apply(lambda x: x[layer][strategy]).to_numpy()
        test_x = features.filter(items=test_idx, axis=0)[f"{obj}_fg_tokens_act"].apply(lambda x: x[layer][strategy]).to_numpy()
        
        clf_model = base_model(model, n_classes=len(set(labels)))
        print(f"Experiment {exp_n}/{len(params)} {exp_name} - training model l:{layer} o:{obj} s:{strategy}")
        if model == "NN":
            es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20, restore_best_weights=True)
            hist = clf_model.fit(tf.stack(train_x), 
                                 tf.stack(train_y), 
                                 validation_data=(tf.stack(val_x),tf.stack(val_y)),
                                 epochs=epochs, 
                                 batch_size=128, 
                                 callbacks=[es],
                                 workers=6,
                                 use_multiprocessing=False,
                                 verbose=0)
            # save model
            clf_model.save(os.path.join(EXP_FOLDER, exp_name, f"model-l{layer}-o{obj}-s{strategy}"))
            histories[p] = hist
            # Save raw loss and acc from "hist" object to recreate plots
            with open(f"{EXP_FOLDER}/{exp_name}/model_l-{layer}_o-{obj}_t-{strategy}_history.pickle", 'wb') as handle:
                pickle.dump(hist.history, handle)
            print("Evaluating model...")                
            test_scores[p] = clf_model.evaluate(tf.stack(test_x), 
                                                tf.stack(test_y), 
                                                batch_size=128)
            #TODO: Evaluation per class using saved preds
            preds = clf_model.predict(tf.stack(test_x), batch_size=128)
            np.save(os.path.join(EXP_FOLDER, exp_name, f"preds_l-{layer}_o-{obj}_t-{strategy}.npy"), preds)
            y_pred = np.argmax(preds, axis=1)
            matrix = confusion_matrix(test_labels, y_pred)
            class_scores[p] = matrix.diagonal()/matrix.sum(axis=1)
            print("Done...")
        else: 
            clf_model.fit(np.stack(train_x), train_y)
            y_pred = clf_model.predict(np.stack(test_x))
            test_scores[p] = metrics.accuracy_score(test_y, y_pred)

    if model == "NN":
        plot_training_curves(histories, f"{EXP_FOLDER}/{exp_name}/training_curves.png")
    save_test_scores(test_scores, exp_name, f"{EXP_FOLDER}/{exp_name}/test_scores.csv")
    save_test_scores(class_scores, exp_name, f"{EXP_FOLDER}/{exp_name}/class_scores.csv", class_labels=unique_label_names)

if __name__ == "__main__":
    # dataset_folder = "features"
    # for feat_group_name in os.listdir(dataset_folder):        
    #     if "features-" in feat_group_name:
    #         print(f"feature set: {feat_group_name}")
    #         exp_name = f"exp_{feat_group_name[9:]}" 
    #         os.makedirs(os.path.join(EXP_FOLDER, exp_name), exist_ok=True)
    #         features, labels = prepare_dataset(f"{dataset_folder}/{feat_group_name}")
    #         run_experiments(features=features, 
    #                         labels=features["labels"], 
    #                         exp_name=exp_name,
    #                         models=["NN","SVM"])

    dataset_folder = "feat_two_objects/features-mask-4-main_thr-0-sec_thr-0/"
    # dataset_folder = "features_dining-table/features-mask-4-main_thr-0-sec_thr-0/"
    exp_name = f"exp_full_{dataset_folder[26:-1]}_attn_filter"
    # exp_name = f"exp_dining_{dataset_folder[31:-1]}_sgd2"
    os.makedirs(os.path.join(EXP_FOLDER, exp_name), exist_ok=True)
    features, labels = prepare_dataset(dataset_folder)
    print(labels)
    run_experiments(features=features, 
                    labels=features["labels"], 
                    unique_label_names = labels,     
                    exp_name=exp_name,
                    epochs=200,
                    models=["NN"])