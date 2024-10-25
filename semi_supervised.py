import streamlit as st
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.semi_supervised import SelfTrainingClassifier, LabelSpreading
from sklearn import metrics

@st.cache_data
def load_corpus():
    """
    Load the corpus from a CSV file
    """
    corpus = pd.read_csv("abstracts.csv")
    # Combine the title and abstract
    corpus['Content'] = corpus.apply(lambda row: 
        f"{row['Article Title']}. {row['Abstract']}", axis=1)
    # Change column names to lowercase
    corpus.columns = [col.lower() for col in corpus.columns]
    # Sections of the corpus we want to keep
    return corpus[['authors', 'article title', "source title", 'abstract', 
        'content', 'publication year']]

def load_labels():
    if not os.path.exists("abstracts-labels.csv"):
        st.info("No labels found. Please label at least 3 abstracts first.", icon="⚠️")
        st.stop()
    return pd.read_csv("abstracts-labels.csv")

st.sidebar.title("Semi-supervised learning")
st.markdown("Welcome to the semi-supervised learning playground. " +
    "This app demonstrates the use of semi-supervised learning to " +
    "learn the remaining labels in a partially labeled dataset.")
st.markdown("You can use this app to:")
st.markdown("* View the corpus after the initial labels have been added (-1 means unlabeled)")
st.markdown("* Inspect the term-document matrix of the relative frequencies of the terms in the documents")
st.markdown("* Evaluate the performance of the model through a confusion matrix")
st.markdown("* View the labels predicted for the unlabeled documents")
st.markdown("* Add more labels to unlabeled documents")
st.markdown("The last step illustrates a typical use case. Suppose you want to screen papers for a literature " +
    "review. You can keep labeling documents until you are satisfied with the performance of the model.")

corpus = load_corpus()
labels = load_labels()

with st.status("Merge labels with corpus"):
    # Merge the labels with the corpus
    corpus['label'] = -1 # initialize all labels to -1 (unlabeled)
    # Set the labels for the labeled abstracts
    for _, row in labels.iterrows():
        corpus.loc[row['id'], 'label'] = row['label']

if st.sidebar.checkbox("Show corpus"):
    st.header("Corpus")
    st.markdown("This corpus consists of abstracts from the Web of Science. " +
        "Examine the labels assigned to the papers.")
    st.markdown("What is the meaning of each label (0, 1, or -1)?")
    st.dataframe(corpus)

# Create a term-document matrix
# The term-document matrix is a dataframe with the words as rows and the documents as columns
# Create a pipeline to vectorize the abstracts
preprocessor = Pipeline(steps=[
    ('vectorizer', CountVectorizer(stop_words='english', max_features=50, min_df=5)),
    ('tfidf', TfidfTransformer())])

with st.status("Compute term-document matrix"):
    tfidf = preprocessor.fit_transform(corpus['content'])
    vectorizer = preprocessor.named_steps['vectorizer']
    vocab = vectorizer.get_feature_names_out()
    tdm = pd.DataFrame(tfidf.toarray().T, index=vocab)

if st.sidebar.checkbox("Show term-document matrix"):
    st.header("Term-document matrix")
    st.markdown("The rows are words (terms) and the columns are documents.")
    st.markdown("What are the 3 most important terms in the corpus? Find them by sorting the columns.")
    st.dataframe(tdm)

# Self-training requires a base estimator
base_estimator = LogisticRegression(penalty='l2', class_weight='balanced')

# Self-training algorithm and pipeline
classifier = SelfTrainingClassifier(base_estimator)
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', classifier)])

# We could alternatively use label spreading
# classifier = LabelSpreading()
# model = Pipeline(steps=[
#     ('vectorizer', CountVectorizer(stop_words='english', min_df=2, max_features=100)),
#     ('tfidf', TfidfTransformer()),
#     ('to_dense', FunctionTransformer(lambda x: x.todense(), accept_sparse=True)),
#     ('classifier', classifier)])

# Apply the pipeline to the corpus
with st.status("Predict missing labels"):
    model.fit(corpus['content'], corpus['label'])
    corpus['predicted'] = model.predict(corpus['content'])

if st.sidebar.checkbox("Confusion matrix"):
    st.header("Confusion matrix")
    st.markdown("What does this confusion matrix tell you about the performance of the model?")

    # Get indices of labeled abstracts
    labeled = corpus['label'] != -1
    # To compute the confusion matrix, we need the true labels and the predicted labels
    # However, we can only include the labeled abstracts in the evaluation (why?)
    y_test = corpus[labeled]['label']
    y_pred = corpus[labeled]['predicted']

    # Confusion matrix
    conf_matrix = metrics.confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', ax=ax, cmap=plt.cm.Blues)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion matrix')
    ax.yaxis.set_ticklabels(['No', 'Yes'])
    st.pyplot(fig)

if st.sidebar.checkbox("Predicted labels"):
    st.header("Predicted labels")
    st.markdown("These are the labels predicted by the model for the unlabeled abstracts. " +
        "The confusion matrix (see above) will give you a sense of how reliable those labels are. ")
    st.dataframe(corpus)

    # Save the documents with positive labels
    corpus[corpus['predicted'] == 1].to_csv("abstracts-positive.csv", index=False)    

if st.sidebar.checkbox("Add labels"):
    # Note: The following code is a bit more involved than the code so far
    # However, the basic logic can be described in a few steps:
    # 1. Find the least certain predictions
    # 2. Show the abstracts with the least certain predictions
    # 3. Ask for labels for those abstracts
    st.header("Add labels")
    # Get indices of the unlabeled abstracts
    unlabeled = corpus['label'] == -1
    number_of_labeled = len(corpus) - unlabeled.sum()
    st.markdown(f"You labeled {number_of_labeled} of {len(corpus)} (or {100 * number_of_labeled / len(corpus):.0f}%) of abstracts.")
    with st.status("Find the least certain predictions"):
        # Check how certain the model is about the predicted labels
        # Instead of the predicted labels, we can also get the probabilities
        # that the model assigns to each class
        y_pred_proba = model.predict_proba(corpus[unlabeled]['content'])
        # Get the indices of the three least certain predictions
        # Theses are predictions with a probability close to 0.5 for both classes
        indices = np.argsort(y_pred_proba.max(axis=1))[:3]
        # Show the probabilities
        st.write(y_pred_proba[indices])
    with st.form("label-abstracts"):
        st.markdown("Label these abstracts")
        for i in indices:
            abstract = corpus.iloc[i]
            st.markdown(f"**{abstract['article title']}**")
            st.markdown(abstract['content'])
            # Ask for the label (is this abstract relevant?)
            predicted_label = y_pred_proba[i].argmax()
            # Add a new row with the id and the predicted label
            assigned_label = st.toggle("Relevant", predicted_label, key=f"label-{i}")
            # force label to 0 or 1 (otherwise it is False or True)
            labels.loc[i] = [i, int(assigned_label)]
        submit = st.form_submit_button("Submit")
    if submit:
        # Save the updated labels
        labels.to_csv("abstracts-labels.csv", index=False)
        st.success("Labels saved")
        # Rerun the app to update the model
        st.rerun()