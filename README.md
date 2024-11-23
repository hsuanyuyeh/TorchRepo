ConvNets - Implementation of CNN in VGG structure\
\
Diffusion - Implementation of Diffusion model with forward process (noise scheduler) + backward process (UNet)\
\
RAG - Implementation of simple RAG architecture with langchain and chroma database (python3 Create_chromaDB.py | python3 RAG_query "How to improve muscle strength?")\
\
Transformer - Implementation of standard transforsmer with encoder and decoder. Encoder is (SelfAttention + FeedForward) times number of layers. Decoder is (SelfAttention + (SelfAttention + FeedForward)) times number of layers. SelfAttenion computes 3 vectors Query(Q), Key(K), Value(V).\
\
LoRA - Implementation of Low rank adaption to fine tune a NN model. Instead of train the whole network with params = d*k as 5,000,000 (d = 1000, k=5000), using rank = 2, we get (d*r)+(r*k) = 2000*10000 = 12,000 params.