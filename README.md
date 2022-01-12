# Stock-Prediction-

In this project, we aim to enhance the prediction of stock market movements using sentiment analysis and deep learning.
We divide the effort in this project into four phases. In the first part, we aim to find as much textual data in tweets, comments, etc., as possible. We then process, transform and structure this data so that our models can be trained on it. During the second phase,  pre-trained language models are used to generate sentence-level embeddings for each of the samples in our dataset and save these embeddings on disk. In the third part, a capable classifier is trained to take in the embeddings and predict sentiments. We also aggregate the predicted sentiments to generate a single number indicating how positive the sentiment has been for that stock on that particular day. We save these predicted and aggregated sentiments for each stock symbol and day on disk.
Finally, in the fourth phase, we extract price fluctuations for each stock symbol in each day and compute technical features. Appending the new technical features to the sentiment predictions, we then find the best features and train various hybrid deep learning models to take in these features for a window size before the current day and predict the stock price movement for the next day. 
In this project, we have tested our approaches to a total of 24 Nasdaq stocks. Moreover, the results and methodology are available in the report section.

Here is a youtube link for this project: https://youtu.be/D6BLZUh3QHY

This Projct is developed by: Sepehr Asgarian and Rouzbeh MeshkinNejad

# Refrences
@article{lecun1995convolutional,
  title={Convolutional networks for images, speech, and time series},
  author={LeCun, Yann and Bengio, Yoshua and others},
  journal={The handbook of brain theory and neural networks},
  volume={3361},
  number={10},
  pages={1995},
  year={1995}
}

@article{nalluri2019weather,
  title={Weather prediction using clustering strategies in machine learning},
  author={Nalluri, Sravani and Ramasubbareddy, Somula and Kannayaram, G},
  journal={Journal of Computational and Theoretical Nanoscience},
  volume={16},
  number={5-6},
  pages={1977--1981},
  year={2019},
  publisher={American Scientific Publishers}
}
@article{wang2019application,
  title={Application of a long short-term memory neural network: a burgeoning method of deep learning in forecasting HIV incidence in Guangxi, China},
  author={Wang, G and Wei, W and Jiang, J and Ning, C and Chen, H and Huang, J and Liang, B and Zang, N and Liao, Y and Chen, R and others},
  journal={Epidemiology \& Infection},
  volume={147},
  year={2019},
  publisher={Cambridge University Press}
}
@article{bonakdari2020reliable,
  title={A reliable time-series method for predicting arthritic disease outcomes: New step from regression toward a nonlinear artificial intelligence method},
  author={Bonakdari, Hossein and Pelletier, Jean-Pierre and Martel-Pelletier, Johanne},
  journal={Computer methods and programs in biomedicine},
  volume={189},
  pages={105315},
  year={2020},
  publisher={Elsevier}
}
@article{derakhshan2019sentiment,
  title={Sentiment analysis on stock social media for stock price movement prediction},
  author={Derakhshan, Ali and Beigy, Hamid},
  journal={Engineering applications of artificial intelligence},
  volume={85},
  pages={569--578},
  year={2019},
  publisher={Elsevier}
}

@article{obthong2020survey,
  title={A survey on machine learning for stock price prediction: algorithms and techniques},
  author={Obthong, Mehtabhorn and Tantisantiwong, Nongnuch and Jeamwatthanachai, Watthanasak and Wills, Gary},
  year={2020}
}

@article{jaggi2021text,
  title={Text Mining of Stocktwits Data for Predicting Stock Prices},
  author={Jaggi, Mukul and Mandal, Priyanka and Narang, Shreya and Naseem, Usman and Khushi, Matloob},
  journal={Applied System Innovation},
  volume={4},
  number={1},
  pages={13},
  year={2021},
  publisher={Multidisciplinary Digital Publishing Institute}
}

@article{devlin2018bert,
  title={Bert: Pre-training of deep bidirectional transformers for language understanding},
  author={Devlin, Jacob and Chang, Ming-Wei and Lee, Kenton and Toutanova, Kristina},
  journal={arXiv preprint arXiv:1810.04805},
  year={2018}
}

@inproceedings{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and Kaiser, {\L}ukasz and Polosukhin, Illia},
  booktitle={Advances in neural information processing systems},
  pages={5998--6008},
  year={2017}
}

@article{nguyen2015sentiment,
  title={Sentiment analysis on social media for stock movement prediction},
  author={Nguyen, Thien Hai and Shirai, Kiyoaki and Velcin, Julien},
  journal={Expert Systems with Applications},
  volume={42},
  number={24},
  pages={9603--9611},
  year={2015},
  publisher={Elsevier}
}

@inproceedings{jeong2018algorithm,
  title={An algorithm for supporting decision making in stock investment through opinion mining and machine learning},
  author={Jeong, Yujin and Kim, Sunhye and Yoon, Byungun},
  booktitle={2018 Portland International Conference on Management of Engineering and Technology (PICMET)},
  pages={1--10},
  year={2018},
  organization={IEEE}
}

@book{alpaydin2020introduction,
  title={Introduction to machine learning},
  author={Alpaydin, Ethem},
  year={2020},
  publisher={MIT press}
}

@inproceedings{nair2010rectified,
  title={Rectified linear units improve restricted boltzmann machines},
  author={Nair, Vinod and Hinton, Geoffrey E},
  booktitle={Icml},
  year={2010}
}

@misc{berthuggingface, title={Huggingface BERT webpage}, url={https://huggingface.co/docs/transformers/master/en/model_doc/bert#overview}, journal={Huggingface Transformers}} 

@article{chong2017deep,
  title={Deep learning networks for stock market analysis and prediction: Methodology, data representations, and case studies},
  author={Chong, Eunsuk and Han, Chulwoo and Park, Frank C},
  journal={Expert Systems with Applications},
  volume={83},
  pages={187--205},
  year={2017},
  publisher={Elsevier}
}
@inproceedings{lam2019gaussian,
  title={Gaussian process lstm recurrent neural network language models for speech recognition},
  author={Lam, Max WY and Chen, Xie and Hu, Shoukang and Yu, Jianwei and Liu, Xunying and Meng, Helen},
  booktitle={ICASSP 2019-2019 IEEE international conference on acoustics, speech and signal processing (ICASSP)},
  pages={7235--7239},
  year={2019},
  organization={IEEE}
}
@article{hochreiter1997long,
  title={Long short-term memory},
  author={Hochreiter, Sepp and Schmidhuber, J{\"u}rgen},
  journal={Neural computation},
  volume={9},
  number={8},
  pages={1735--1780},
  year={1997},
  publisher={MIT Press}
}
@article{sangeetha2020sentiment,
  title={Sentiment analysis of student feedback using multi-head attention fusion model of word and context embedding for LSTM},
  author={Sangeetha, K and Prabha, D},
  journal={Journal of Ambient Intelligence and Humanized Computing},
  pages={1--10},
  year={2020},
  publisher={Springer Berlin Heidelberg}
}
