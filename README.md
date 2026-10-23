# Netflix-Movies-shows-clustering-

Netflix Movies & TV Shows Clustering

Unsupervised Machine Learning | Data Analysis | Text Mining

1. Project Overview

The Netflix Movies & TV Shows Clustering project is a Data Science and Machine Learning project that analyzes Netflix's content catalogue using Exploratory Data Analysis (EDA) and Unsupervised Machine Learning techniques.

Netflix contains a wide variety of movies and TV shows across different genres, countries, release years, audience ratings, and content categories. Analyzing this large collection can help identify patterns and similarities among different titles.

This project uses Python to clean and analyze the Netflix dataset, visualize important trends, extract meaningful text-based features, and apply clustering algorithms to group content based on similarities in content type, genres, and descriptions.

The project implements TF-IDF, K-Means Clustering, and Agglomerative Clustering to explore the structure of the Netflix content catalogue. It also uses the Elbow Method, Dendrogram Visualization, and Silhouette Score to examine clustering behaviour and evaluate the separation between groups.

2. Problem Statement

Netflix offers a large collection of movies and television shows with different genres, descriptions, countries, ratings, and release years.

Manually analyzing thousands of titles to identify similarities and patterns is difficult. A data-driven approach can help organize this information and explore groups of titles with similar textual characteristics.

The main problem addressed in this project is to identify patterns and discover groups of similar Netflix movies and TV shows using Unsupervised Machine Learning techniques.

The project transforms selected textual information into numerical features and applies clustering algorithms to explore relationships between titles without using predefined target labels.

3. Project Objectives

The main objectives of this project are:

- To explore the Netflix Movies and TV Shows dataset.
- To identify missing values and duplicate records.
- To clean and preprocess the dataset.
- To perform Exploratory Data Analysis (EDA).
- To analyze the distribution of movies and TV shows.
- To examine country-wise content distribution.
- To analyze content ratings and popular genres.
- To visualize content added to Netflix over time.
- To combine content type, genres, and descriptions into text-based features.
- To convert textual data into numerical representations using TF-IDF.
- To apply K-Means Clustering to group similar titles.
- To apply Agglomerative Clustering for hierarchical grouping.
- To visualize clustering results using the Elbow Method and Dendrogram.
- To evaluate clustering performance using the Silhouette Score.
- To summarize cluster characteristics and export the processed dataset.

4. Dataset Description

The project uses a Netflix Movies and TV Shows dataset stored in CSV format.

The dataset contains information about Netflix titles, including:

- Show ID
- Content Type
- Title
- Director
- Cast
- Country
- Date Added
- Release Year
- Content Rating
- Duration
- Genre Categories
- Description

The Python code expects the dataset file to be named:

"NETFLIX MOVIES AND TV SHOWS CLUSTERING.csv"

The exact number of records and available columns depend on the dataset version used.

Data Preprocessing

The dataset is loaded into a Pandas DataFrame. Initial inspection is performed to examine the dataset dimensions, column information, missing values, and duplicate records.

Missing values in text columns are replaced with the placeholder "Unknown", and leading or trailing spaces are removed from text values.

The "date_added" column is converted into a datetime format, and a new "year_added" feature is created to analyze the content-added trend.

These preprocessing steps help prepare the dataset for exploratory analysis and machine learning.

5. Technologies and Libraries Used

Technology| Purpose
Python| Main programming language
Pandas| Data manipulation and preprocessing
NumPy| Numerical operations
Matplotlib| Data visualization
Seaborn| Statistical data visualization
Scikit-learn| Machine learning and text feature extraction
TF-IDF Vectorizer| Text-to-numerical feature transformation
K-Means| Unsupervised clustering
Agglomerative Clustering| Hierarchical clustering
SciPy| Hierarchical clustering and dendrogram
Google Colab| Project development and execution
GitHub| Source code management and project submission

6. Project Workflow

The project follows a structured Data Science workflow:

1. Dataset loading and inspection.
2. Missing-value and duplicate-value analysis.
3. Data cleaning and preprocessing.
4. Exploratory Data Analysis.
5. Data visualization.
6. Text feature engineering.
7. TF-IDF vectorization.
8. K-Means Clustering.
9. Agglomerative Clustering.
10. Cluster analysis and interpretation.
11. Elbow Method visualization.
12. Dendrogram visualization.
13. Silhouette Score evaluation.
14. Exporting the processed dataset.
15. Final project summary and conclusion.

7. Exploratory Data Analysis (EDA)

Exploratory Data Analysis is performed to understand the distribution and characteristics of Netflix content.

7.1 Movies vs. TV Shows

A bar chart is created to compare the number of movies and TV shows in the dataset.

The analysis in the notebook reports that movies outnumber TV shows in the dataset version used. This comparison provides an initial overview of the catalogue's content composition.

7.2 Content Added Over Time

The "date_added" field is converted into datetime format, and the "year_added" feature is extracted.

A time-series visualization compares the number of movies and TV shows added during different years.

This analysis helps explore changes in the catalogue over time. The year added to Netflix is different from the original release year of a title.

7.3 Country-Wise Content Distribution

The country information is analyzed to identify countries frequently associated with Netflix titles.

Since a title can be associated with multiple countries, country entries are separated before counting their occurrences.

A visualization displays the top ten country entries in the dataset.

7.4 Content Rating Analysis

The rating column is analyzed to identify the most frequently occurring content ratings.

A bar chart visualizes the distribution of ratings and helps describe the audience categories represented in the dataset.

7.5 Genre Analysis

The "listed_in" column contains genre and category information.

The genre values are separated into individual entries to identify the most frequently listed categories.

The analysis highlights commonly occurring categories such as International Movies, Dramas, Comedies, and Documentary-related content.

Because a title can belong to multiple genres, genre counts may overlap.

8. Text Feature Engineering

Text feature engineering prepares textual information for machine learning algorithms.

In this project, three important fields are combined:

- Content Type
- Listed Genres
- Description

These fields are combined into a new feature called "content_text".

The combined text contains descriptive information about each title and provides the input for TF-IDF vectorization.

9. TF-IDF Vectorization

TF-IDF (Term Frequency–Inverse Document Frequency) is used to convert textual data into numerical features.

Clustering algorithms require numerical representations instead of raw text. TF-IDF assigns weights to terms based on their frequency within a document and across the collection.

The project uses "TfidfVectorizer" with the following settings:

- Maximum features: 3,000
- English stop-word removal
- Input: Combined content type, genres, and descriptions

The resulting TF-IDF matrix is used as the input for the clustering algorithms.

TF-IDF helps identify similarities in word usage, but it does not fully understand the meaning or context of a movie or TV show's description.

10. K-Means Clustering

K-Means is an Unsupervised Machine Learning algorithm that groups data points into a specified number of clusters.

In this project, K-Means is applied to the TF-IDF feature matrix with five clusters.

Each Netflix title receives a cluster label based on the numerical representation of its textual features.

The resulting clusters are analyzed to understand their size, content-type distribution, common genres, and content ratings.

Cluster Analysis

The project examines:

- The number of titles in each cluster.
- The distribution of movies and TV shows.
- The most common genres in each cluster.
- The most frequent content ratings in each cluster.

This analysis helps explore patterns in the Netflix catalogue.

Cluster labels are generated by the algorithm and do not represent official Netflix categories.

11. Agglomerative Clustering

Agglomerative Clustering is a hierarchical Unsupervised Machine Learning algorithm.

It begins with individual observations and gradually merges them into larger groups based on their feature similarities.

In this project, a sample of 1,000 records is selected from the dataset to reduce computational requirements.

Agglomerative Clustering is then applied to the sample with five clusters and Ward linkage.

The resulting cluster assignments are examined to understand the hierarchical grouping behaviour of the selected data.

12. Elbow Method

The Elbow Method is used to investigate a suitable number of clusters for K-Means.

The algorithm is fitted using different cluster counts, from K = 2 to K = 10, and the corresponding inertia values are calculated.

The results are displayed in a line chart.

An elbow-shaped bend may indicate a useful balance between cluster compactness and model complexity.

In the notebook's analysis, the curve does not show a clearly defined elbow. Therefore, selecting five clusters is an exploratory choice rather than a definitive optimum.

13. Dendrogram Visualization

A dendrogram is generated to visualize the hierarchical merging process used in hierarchical clustering.

The project creates a dendrogram from a subset of 200 feature vectors taken from the sample.

The visualization shows how groups merge at different distances and helps explore the hierarchical structure of the selected records.

The dendrogram is used as an analytical visualization and does not independently establish the best number of clusters.

14. Model Evaluation Using Silhouette Score

The Silhouette Score is used to evaluate how well observations fit within their assigned clusters compared with other clusters.

The project calculates the score for K-Means and Agglomerative Clustering using cosine distance on the selected sample.

The notebook documents the following approximate results:

Algorithm| Silhouette Score
K-Means| 0.0479
Agglomerative Clustering| 0.0142

These values are the results documented in the notebook and may vary with dataset changes or modifications to the code.

Both scores are close to zero, indicating weak separation between clusters under the current feature representation and evaluation setup.

K-Means performs slightly better in this documented comparison, although further feature engineering and model experimentation may improve the results.

15. Data Export and Final Output

After data preprocessing and clustering, the processed dataset is exported to a CSV file named:

"Netflix_Clustering_Results.csv"

The exported dataset is intended to contain the processed data, engineered text feature, and K-Means cluster assignments.

The output can be used for further analysis, reporting, and future development.

The file is generated when the script completes successfully.

16. Key Findings

Based on the analysis documented in the notebook:

- Movies are more numerous than TV shows in the dataset version analyzed.
- Country, rating, and genre fields provide useful descriptive insights into the catalogue.
- TF-IDF converts selected textual information into numerical features for clustering.
- K-Means identifies five groups of titles based on the selected text representation.
- Agglomerative Clustering provides an alternative hierarchical grouping approach.
- Cluster summaries reveal differences in content-type and genre distributions.
- The Elbow Method does not identify a clearly defined optimal cluster count.
- The documented Silhouette Scores indicate that the current clusters are not strongly separated.

These findings demonstrate how data analysis and unsupervised machine learning can be applied to explore a large content catalogue.

17. Installation and Execution

Prerequisites

- Python
- pip
- The required Python libraries
- The Netflix dataset CSV file

Step 1: Clone the Repository

git clone <YOUR_GITHUB_REPOSITORY_URL>
cd <YOUR_REPOSITORY_FOLDER>

Replace the placeholders with the actual repository URL and folder name.

Step 2: Install Dependencies

pip install -r requirements.txt

Step 3: Add the Dataset

Place the dataset CSV file in the project directory and ensure the dataset path in the Python script matches its location.

The current script uses an absolute path:

df = pd.read_csv('/NETFLIX MOVIES AND TV SHOWS CLUSTERING.csv')

For portability, the path can be changed to a relative path when the CSV is stored in the project directory:

df = pd.read_csv('NETFLIX MOVIES AND TV SHOWS CLUSTERING.csv')

Step 4: Run the Project

python netflix_movies_tv_shows_clustering.py

The script is designed to display visualizations, perform clustering analysis, and generate the processed CSV output.

Note: The dataset path and Python code should be checked before execution in a new environment.

18. Limitations

- The quality of results depends on the dataset and preprocessing choices.
- TF-IDF represents textual patterns but does not fully capture semantic meaning.
- The choice of five clusters is exploratory.
- The documented Silhouette Scores indicate weak cluster separation.
- Agglomerative Clustering is applied to a sample rather than the entire dataset.
- Missing date values are filled with placeholder values in the supplied code; those placeholders are not confirmed Netflix dates.
- The dataset path may need to be changed when running the project on another computer.
- The clustering results should not be treated as definitive recommendations.

19. Future Improvements

The project can be improved through the following enhancements:

- Experiment with different numbers of clusters.
- Tune clustering algorithm parameters.
- Compare alternative text vectorization techniques.
- Explore additional clustering algorithms.
- Improve missing-value handling.
- Add validation for required dataset columns.
- Develop a title-search feature to inspect titles in the same cluster.
- Create an interactive dashboard for cluster exploration.
- Save visualizations and evaluation results for reporting.
- Improve reproducibility by documenting package versions and execution steps.

20. Conclusion

The Netflix Movies & TV Shows Clustering project demonstrates the application of Python, Exploratory Data Analysis, text feature engineering, and Unsupervised Machine Learning to a real-world content catalogue.

The project uses TF-IDF to represent content type, genres, and descriptions as numerical features and applies K-Means and Agglomerative Clustering to explore patterns among Netflix titles.

The EDA visualizations provide insights into content types, countries, ratings, genres, and content-added trends. Cluster analysis further helps examine differences in the distribution of content types and genres.

The documented Silhouette Scores show that the current clustering configurations produce weakly separated groups, highlighting the importance of model evaluation and continued experimentation.

Overall, this project provides practical experience with data preprocessing, visualization, text vectorization, clustering algorithms, model evaluation, and data export. It also establishes a foundation for future improvements in content analysis and recommendation-oriented applications.

21. Author

Pratik Singha Roy

Data Science | Python | Machine Learning | Data Analysis

22. Disclaimer

This project is intended for educational and portfolio purposes. Netflix and its related trademarks belong to their respective owners. This project is not affiliated with, sponsored by, or endorsed by Netflix.
