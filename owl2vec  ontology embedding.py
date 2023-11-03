import rdflib
import networkx as nx
from owlready2 import get_ontology, default_world
from owl2vec_star.rdf2vec.learner import Rdf2VecLearner
from owl2vec_star.rdf2vec import Rdf2VecTransform, KG
from sklearn.metrics import accuracy_score, recall_score, f1_score

# Define functions for ontology parsing and graph creation
def parse_ontology(ontology_path):
    g = rdflib.Graph()
    g.parse(ontology_path)
    return g

def create_graph_from_ontology(ontology_path):
    g = parse_ontology(ontology_path)
    graph = nx.Graph()
    
    for subject, predicate, obj in g:
        graph.add_node(subject)
        graph.add_node(obj)
        graph.add_edge(subject, obj, predicate=predicate)
    
    return graph

def main():
    # Paths to the ontologies
    cmt_ontology_path = "cmt.owl"
    ekaw_ontology_path = "ekaw.owl"

    # Create graphs from the ontologies
    cmt_graph = create_graph_from_ontology(cmt_ontology_path)
    ekaw_graph = create_graph_from_ontology(ekaw_ontology_path)

    # Load ontologies using owlready2
    onto_cmt = get_ontology("file://" + cmt_ontology_path).load()
    onto_ekaw = get_ontology("file://" + ekaw_ontology_path).load()

    # Create RDF2Vec learners
    learners = {
        "cmt": Rdf2VecLearner(onto_cmt.world),
        "ekaw": Rdf2VecLearner(onto_ekaw.world),
    }

    # Train RDF2Vec models
    for name, learner in learners.items():
        learner.learn("TransE", kg=KG(name=name, owlready_graph=onto_cmt))
        learner.learn("DistMult", kg=KG(name=name, owlready_graph=onto_cmt))
    
    # Transform RDF2Vec embeddings
    transformers = {
        "cmt": Rdf2VecTransform(learners["cmt"].model, "TransE"),
        "ekaw": Rdf2VecTransform(learners["ekaw"].model, "TransE"),
    }

    cmt_embeddings = transformers["cmt"].transform(onto_cmt.classes())
    ekaw_embeddings = transformers["ekaw"].transform(onto_ekaw.classes())

    # Matching similar vectors using owl2vec_star (you can use a suitable threshold)
    threshold = 0.8
    matches = []
    for i, cmt_embedding in enumerate(cmt_embeddings):
        for j, ekaw_embedding in enumerate(ekaw_embeddings):
            similarity = cmt_embedding.similarity(ekaw_embedding)
            if similarity >= threshold:
                matches.append((i, j))

    # Calculate accuracy, recall, and F-measure
    true_positives = len(matches)
    false_positives = len(cmt_embeddings) - len(matches)
    false_negatives = len(ekaw_embeddings) - len(matches)

    accuracy = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f_measure = 2 * (accuracy * recall) / (accuracy + recall)

    print("Accuracy: {:.2f}".format(accuracy))
    print("Recall: {:.2f}".format(recall))
    print("F-measure: {:.2f}".format(f_measure))

if __name__ == "__main__":
    main()
