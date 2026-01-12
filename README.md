Project Title
A High-Fidelity RAG Pipeline for Narrative Veracity and Logical Consistency

Project Description:
Our project implements an end-to-end Retrieval-Augmented Generation (RAG) system designed to audit the logical consistency of character backstories against dense, unstructured literary narratives. Utilizing Pathway for high-throughput stream processing, we engineered a hybrid ingestion pipeline that pairs semantic-aware chunking with stateful metadata inheritance to eliminate context fragmentation. By enforcing a  reasoning framework on a Mistral core, the system suppresses model hallucinations through negative-constraint prompting. The final architecture delivers precise binary consistency judgments (0/1) backed by transparent, chapter-specific rationales, ensuring scalability across massive datasets without compromising logical fidelity.

Prerequisites
This project is designed to run on Linux or Windows Subsystem for Linux (WSL). Ensure you have the necessary environment set up before proceeding.

Getting Started
 Please follow these steps to set up the environment and run the code:

1. Prepare the Data
Before running the execution script, ensure the directory structure is correct:

Add the Dataset: Place your input data inside a folder named Dataset/ in the root directory.

Clean Previous Runs: To re-run the code, you must delete the results.csv file to ensure a clean run and prevent data overlap.

2. Execution

Open your terminal (WSL/Linux) and run the following command from the project root:
bash run.sh