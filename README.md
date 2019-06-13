# TrabFinal_TEIA
Desenvolvimento de uma IA para descobrir o gênero de músicas

## Dataset
Foi utilizado um conjunto de dados pronto (conjunto de treino e teste devidamente atrelados as suas respectivas labels, com extração de amostras para cada música, ex: 30s por música, já convertidos para mono). Podendo futuramente, como implementação extra, a obtenção de um conjunto próprio de dados brutos, podendo ser obtidos via plataformas de streaming (Spotify, iTunes, Last.fm entre outros) ou banco de dados de áudio que possuem em seus metadados o gênero musical (Albuns pessoais, Million Song Dataset (MSD), entre outros).

O conjunto de dados escolhido é bastante comum e famoso para treino e predições de gêneros musicais, o GTZAN[1], usado também em " Musical genre classification of audio signals " by G. Tzanetakis and P. Cook in IEEE Transactions on Audio and Speech Processing 2002.[2]

Este conjunto de dados consistem em 1000 áudios de 30 segundos cada, contendo 10 gêneros diferentes, cada um contendo 100 áudios. Todos os áudios possuem frequência máxima de 22050Hz, todas em mono 16-bits no formato '.au'. Cada conjunto de áudio esta contido em uma pasta diferente que representam os gêneros musicais: blue, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock.

O conjunto de dados foi criado gradualmente pela autora[1] entre 2000 e 2001, juntando músicas de diversas fontes (CDs, rádios, gravações pelo microfone), todas de sua coleção particular, não possuindo direitos autorais nem os nomes das músicas, entretanto apresentando músicas advindas sobe diversas condições e qualidade, o que é muito interessante para o treinamento.

Este dataset possui aproximadamente 1.2GB e pode ser obtido em: http://opihi.cs.uvic.ca/sound/genres.tar.gz

#### Referências
[1] GTZAN Genre Collection. Acessado em 10 de Junho de 2019. Disponivel em: http://marsyas.info/downloads/datasets.html

[2] "Musical genre classification of audio signals " by G. Tzanetakis and P. Cook in IEEE Transactions on Audio and Speech Processing 2002. Acessado em 10 de Junho de 2019. Disponível em: https://ieeexplore.ieee.org/document/1021072

[3] Sturm, Bob L. "An Analysis of the GTZAN Music Genre Dataset" - Aalborg Universitet - 2012. Acessado em 10 de Junho de 2019. Disponível em: https://vbn.aau.dk/ws/portalfiles/portal/74499095/GTZANDB.pdf

### Informações úteis
#### Bibliotecas de Áudio:
- Librosa (Foco em análise de áudio: extração de features (centroide, roll-off e muitas outras), plots (taís como waveform, spectrograma) e escrita de áudio) [4]
- IPython.display.Audio (Realiza leitura, reprodução inclusive dentro do Jupyter notebook)

#### Referências
[1] H. Bahuleyan, "Music Genre Classification using Machine Learning Techniques", 2018. Acessado em 10 de Junho de 2019. Disponível em https://arxiv.org/abs/1804.01149

[2] J. Despois, "Finding the genre of a song with Deep Learning", 2016. Acessado em 10 de Junho de 2019. Disponível em https://medium.com/@juliendespois/finding-the-genre-of-a-song-with-deep-learning-da8f59a61194

[3] M. Lachmish, "Music Genre Classification", 2018. Acessado em 10 de Junho de 2019. Disponível em https://medium.com/@matanlachmish/music-genre-classification-470aaac9833d

[4] P. Pandey, "Music Genre Classification with Python: A Guide to analysing Audio/Music signals in Python", 2018. Acessado em 10 de Junho de 2019. Disponível em https://towardsdatascience.com/music-genre-classification-with-python-c714d032f0d8
