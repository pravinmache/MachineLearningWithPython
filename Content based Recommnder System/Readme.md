# Content based recommender system

Content based recommnder system works based on the genre of the movie and rating provided by the user for movies.

Suppose user has rated movies as following

|Movie Name  | Genres   | Rating | 
|------------- | ------------- | ------|
|Movie A  | Drama, Action, Commedy |  4 |
|Movie B  | Comedy, Adenture | 5 |

We encode movies with one hot encoding schema


|         | Drama | Action  | Comedy | Adventure |
|---------|-------|---------|--------|-----------|
| Movie A | 1     | 1       | 1      | 0         |
| Movie B | 0     | 0       | 1      | 1         |


We calculate weighted feature matrix by multiplying ratings to Weighted genre matrix

|         | Drama | Action  | Comedy | Adventure |
|---------|-------|---------|--------|-----------|
| Movie A | 4     | 4       | 4      | 0         |
| Movie B | 0     | 0       | 5      | 5         |

Create user profile by taking sum along columns

| Drama | Action  | Comedy | Adventure |
|---------|-------|---------|--------|
| 4     | 4       | 9      | 5         |


Create Normalised user profile

| Drama | Action  | Comedy | Adventure |
|-------|---------|--------|-----------|
 | 0.18  | 0.18    | 0.41   | 0.23      |
 
We have following movies to recommend to user with genre

|         | Drama | Action  | Comedy | Adventure |
|---------|-------|---------|--------|-----------|
| Movie C | 1     | 0       | 0      | 1         |
| Movie D | 0     | 1       | 1      | 0         |
| Movie E | 1     | 0       | 0      | 0         |

We will multiply user profile with this matrix and take sum along row

|         | Drama | Action  | Comedy | Adventure | Total 
|---------|-------|---------|--------|-----------| ------|
| Movie C | 0.18  | 0       | 0      | 0.23      | 0.41  |
| Movie D | 0     | 0.18    | 0.41   | 0         | 0.59 |
| Movie E | 0.18  | 0       | 0      | 0         | 0.18 |

This is our recommendation matrix. We can recommend Movie D to user
