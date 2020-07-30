


#import os
#os.chdir("E:\\Bokey\\Excelr Data\\Python Codes\\all_py\\Recommender Systems")
import pandas as pd
#import Dataset 
book = pd.read_csv("file:\\D:\\Data Science\\Excelr\\Assignments\\Assignment\\Recommendation System\\books1.csv")
book.shape #shape
book.columns
book.Author#ratings column


from sklearn.feature_extraction.text import TfidfVectorizer #term frequencey- inverse document frequncy is a numerical statistic that is intended to reflect how important a word is to document in a collecion or corpus

# Creating a Tfidf Vectorizer to remove all stop words
tfidf = TfidfVectorizer(stop_words="english")    #taking stop words from tfid vectorizer 

# replacing the NaN values in overview column with
# empty string
book["Author"].isnull().sum() 
book["Author"] = book["Author"].fillna(" ")

# Preparing the Tfidf matrix by fitting and transforming

tfidf_matrix = tfidf.fit_transform(book.Author)   #Transform a count matrix to a normalized tf or tf-idf representation
tfidf_matrix.shape #12294,46

# with the above matrix we need to find the 
# similarity score
# There are several metrics for this
# such as the euclidean, the Pearson and 
# the cosine similarity scores

# For now we will be using cosine similarity matrix
# A numeric quantity to represent the similarity 
# between 2 movies 
# Cosine similarity - metric is independent of 
# magnitude and easy to calculate 

# cosine(x,y)= (x.y⊺)/(||x||.||y||)

from sklearn.metrics.pairwise import linear_kernel

# Computing the cosine similarity on Tfidf matrix
cosine_sim_matrix = linear_kernel(tfidf_matrix,tfidf_matrix)

# creating a mapping of book name to index number 
book_index = pd.Series(book.index,index=book['Title']).drop_duplicates()


book_index["Clara Callan"]

def get_book_recommendations(Title,topN):
    
   
    #topN = 10
    # Getting the movie index using its title 
    book_id = book_index[Title]
    
    # Getting the pair wise similarity score for all the book's with that 
    # book
    cosine_scores = list(enumerate(cosine_sim_matrix[book_id]))
    
    # Sorting the cosine_similarity scores based on scores 
    cosine_scores = sorted(cosine_scores,key=lambda x:x[1],reverse = True)
    
    # Get the scores of top 10 most similar book's 
    cosine_scores_10 = cosine_scores[0:topN+1]
    
    # Getting the book index 
    book_idx  =  [i[0] for i in cosine_scores_10]
    book_scores =  [i[1] for i in cosine_scores_10]
    
    # Similar movies and scores
    book_similar_show = pd.DataFrame(columns=["Title","Score"])
    book_similar_show["Title"] = book.loc[book_idx,"Title"]
    book_similar_show["Score"] = book_scores
    book_similar_show.reset_index(inplace=True)  
    book_similar_show.drop(["index"],axis=1,inplace=True)
    print (book_similar_show)
    #return (book_similar_show)

    
# Enter your book and number of book's to be recommended 
get_book_recommendations("Classical Mythology",topN=20)
