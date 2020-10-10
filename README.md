# levelr
Node.js &amp; Python machine learning music recommendations that promote independent artists

Levelr is a focused multi-source music recommendation service that promotes independent artists and gives back to marginalized communities.

The features of the service (as of now):
  - Auth using passport.js
  - Feed page w/ RDS connection for music delivery
  - Implementation of Python shells to deliver content thru APIs (Soundcloud, Youtube, more to come)
  - Search functionality in /saved
  - Python keras & Tensorflow algorithm to create recommendation patterns thru Embedded layers w/ randomized weight vectors (n=50)
  
Next steps are implementing effective link scraping to build up sufficient data in RDS to be able to generate effective recommendations, as well as implementing further APIs to allow user connection w/ outside sources that can provide jumpstart data for recommendations.
