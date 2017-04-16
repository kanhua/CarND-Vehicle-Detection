import numpy as np
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.utils import shuffle
from sklearn.pipeline import Pipeline
import pickle
from load_images import load_images

from utils import FeatureExtractor



def find_best_model(all_images,y):

    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        all_images, y, test_size=0.2, random_state=rand_state)

    svc_c=np.array([0,1,1])
    svc_total=np.array([1e-4,1e-3,1e-2])


    color_space_attemps=['RGB','YCrCb','HSV']
    hog_color_space_attempts=['RGB','YCrCb']

    fe=FeatureExtractor(hog_channel='ALL')
    svc=LinearSVC()


    pip_comps = [('fext', fe), ('std', StandardScaler()), ('svc', svc)]
    pip = Pipeline(pip_comps)

    gs=GridSearchCV(pip,dict(fext__color_space=color_space_attemps,
                             fext__hog_color_space=hog_color_space_attempts),verbose=3,cv=5)

    gs.fit(X_train,y_train)

    print(gs.score(X_test,y_test))



def train_best_model(all_images, y):
    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        all_images, y, test_size=0.2, random_state=rand_state)

    svc_d = LinearSVC(C=0.1, tol=1e-3)
    fe = FeatureExtractor(color_space='RGB', hog_channel='ALL', hog_color_space='YCrCb')

    pip_comps = [('fext', fe), ('std', StandardScaler()), ('svc', svc_d)]
    pip = Pipeline(pip_comps)

    # Create an array stack of feature vectors
    print("Len of X: %s" % X_train.shape[0])

    print("number of car: {}".format(y[y == 1].shape[0]))
    print("number of non-car: {}".format(y[y == 0].shape[0]))

    # print('Using spatial binning of:', spatial,
    #      'and', histbin, 'histogram bins')


    print('Feature vector length:', len(X_train[0]))
    # Use a linear SVC
    # Check the training time for the SVC
    t = time.time()
    pip.fit_transform(X_train, y_train)
    t2 = time.time()
    print(round(t2 - t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(pip.score(X_test, y_test), 4))
    # Check the prediction time for a single sample
    t = time.time()
    n_predict = 10
    print('My SVC predicts: ', pip.predict(X_test[0:n_predict]))
    print('For these', n_predict, 'labels: ', y_test[0:n_predict])
    t2 = time.time()
    print(round(t2 - t, 5), 'Seconds to predict', n_predict, 'labels with SVC')

    with open("final_clf.p", 'wb') as fp:
        pickle.dump(pip, fp)


if __name__ == "__main__":
    # Read in car and non-car images

    all_images, y = load_images()

    all_images, y = shuffle(all_images, y, random_state=0)

    train_best_model(all_images, y)

    #find_best_model(all_images,y)
