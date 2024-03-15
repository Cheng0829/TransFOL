# 1

## Extract "Drug-Dose-Target-Response" sub-data set from DrugCombDB.

1) First, connect to DrugCombDB database, some kind of authentication may be required.

2) Using SPARQL query language, construct a query statement to find the data containing the keywords and relationships "drug-dose-target-response".

    ```sparql
    PREFIX dcd: <http://www.drugcombdb.org/ontology#>

    SELECT ?drug ?dose ?target ?response
    WHERE {
    ?exp a dcd:Experiment;
        dcd:hasDrug ?drug;
        dcd:hasDose ?dose;
        dcd:hasTarget ?target;
        dcd:hasResponse ?response.
    }
    ```

    **Explanation**: PREFIX defines the DrugCombDB ontology namespace. The SELECT clause specifies the variables ?drug, ?dose, ?target, and ?response to be selected. The WHERE clause defines a pattern match that looks for experimental data whose instance type is Experiment and has the hasDrug, hasDose, hasTarget, and hasResponse attributes.

    You can add further filters such as.

    ```sparql
    FILTER (?drug IN (<http://www.drugcombdb.org/drug/1234>, ...)) # Filter specific drugs
    FILTER (REGEX(?target, "^HGNC")) # Include only specific gene targets
    ```

3) Set appropriate filters in the query statement, such as selecting only specific types of drugs (e.g., anti-cancer drugs), dose ranges, target types, and so on.

4) Execute the query to get the result set.

5) Check the result set, adjust the query as needed, and repeat the execution until a satisfactory sub-data set is obtained.

6) Export the query results to the desired format, such as CSV, JSON, etc., as the final sub-data set.

## To extract the "Mammalian Symptoms - Human Symptoms - Drugs - Diseases" sub-dataset from Phenomebrowser.

1) Connect to Phenomebrowser database, which may require authentication such as API key.

2) Write code or script to traverse the node and edge data in the database.

    Phenomebrowser uses the graph database Neo4j to store data, you can use the Cypher query language to query. A possible Cypher query is as follows:

    ```cypher
    MATCH path = (:MammalSymptom)-[:RELATED_TO]->(:HumanSymptom)<-[:TREATS]-(drug:Drug)-[:TREATS]->(disease:Disease)
    RETURN path
    ```

    **Explanation**: The MATCH clause defines the pattern to be matched, finding the path from the mammalian symptom to the human symptom, to the drug that treats the symptom, and finally to the disease that the drug treats. The RETURN clause specifies that the full path is returned. You can also add other conditions, such as.

    ```cypher
    WHERE drug.name CONTAINS 'cancer' # Include only drugs used to treat cancer.
    WITH path, disease
    ORDER BY size((disease)-[:GENE_ASSOCIATED_WITH]->()) DESC # Sort in descending order by number of disease-related genes
    ```

3) Extract the data containing the relationship chain "mammalian symptom-human symptom-drug-disease" based on node types and edge labels.

4) Perform necessary cleaning, parsing and pre-processing of the extracted data to obtain structured sub-data sets.

5) Optionally: combine with other data sources to enrich the sub-dataset, such as adding more symptom information, drug annotations and so on.

6) Export the processed data to the desired format, such as graph database, RDF, etc.

