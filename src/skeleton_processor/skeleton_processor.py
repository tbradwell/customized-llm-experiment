# The alogorithm:
# get all previous documents.
# 1 for each document in documents:
# 1.1 for each pargraph in document:
# 1.1.1 save document to vector db: document name, document id, document flat text
# 1.1.2 remove all named entities and replace them with a placeholder.
# 1.1.3 remove all numbers and replace them with a placeholder.
# 1.1.4 create an embedding of the sentence with OpenAI.
# 1.1.5 create a relative position embedding in the entire document. That is the number of the pargraph normalized by the total number of pargraphs.
# 1.1.6 concatenate the embedded sentence and position embedding.
# 1.1.7 Save in vector db: a.  pargraph embedding b. original pargraph text c. clean pargraph text (after 1.1.2-1.1.3) d. pargraph absolute position e. pargraph relative position f. document id e. save font style for each run in the pargraph
# 1.1.8 cluster the pargraphs using the embedding. update the pargraph embedding table with the cluster id.
# 1.1.9 in cluster table add the cluster homogeneity score. 
# 2. Take a random document.
# 3. for each pargraph in document:
# 3.1 take the clean version from db.
# 3.1 create an embedding of the sentence with OpenAI.
# 3.2 find the closet cluster to the embedded sentence.
# 3.3 take the medoid (the member closest to that centroid) of the cluster.
# 3.4 take 2 another 2 farthest members from the centroid that are distinct from each other.
# 3.5 If the current pargraph is mapped to the same cluster as the previous one assign them type 'block'
# 3.6 If the pargraph belongs to a cluster with high homogeneity score then assign it type 'certain'
# 3.7 If the pargraph belongs to a cluster with low homogeneity score then assign it type 'uncertain'
# 3.5.1 save a structure that shows what to replace: original pargraph and location with the clean text (with place holders) of the members
# 4 for each member in the strucutre to replace the pargraph:
# 4.1 take the clean text (with place holders) of the member + the original style of the member.
# 4.2 clean the text from document
# 4.2 If there are multiple pargraphs that map to the same cluster id and are adgacent then the wrapping delimeter is '{%' and '%}' (a delimiter block)
# 4.2.1 add a delimiter block opening '%}'
# 4.2.1 add all the members of the cluster to the block: add the clean text with its saved style to the word document
# 4.2.2 add a delimiter block closing '{%' to after last pargraph.
# 4.2.3 In structure mark as done all pargraphs that are in the same cluster id.
# 4.3 if there arw is only one pargraph that is mapped to a specific ID then add at the beginig 