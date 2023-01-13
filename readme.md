overleaf https://www.overleaf.com/8896864255fpfxxfymkncf

## Run demo:

```bash
pip install -r requirements.txt
streamlit run server.py
```

## 🏃‍♂️ RESULT: [summary](summary.md)

## 🔗 CODE:

- [baseline](Baseline.ipynb)
- [rerank](Rerank.ipynb)
- [bm25](BM25.ipynb)

## 📜 TO-DO:

- [ ] 1. Viết journal cho phần baseline
  - [ ] 1.1 Viết data preprocessing & collect data
  - [ ] 1.2 Viết Experimennt
  - [ ] 1.2 Resuit Compare another
  - [ ] 1.3 Future work
- [ ] 2. Preprocessing wiki data
- [x] 3. Metric used for re-rank answer

  w0 \* retrieval_score + w1 \* predict_score

  w0 + w1= 1

- [x] 4. Run all dataset for base line: WIKI, UIT, SQUAD, FACEBOOK QA
- [x] 6. Chạy context của dataset trước để thử nghiệm mấy task trên cho nhanh
- [x] 7. Context là wiki, chỉ lấy cấu hỏi và gt từ các dataset
- [x] 5. Trường hợp không trả lời được câu hỏi? -> threshold cho cau tra loi null = 0.6 -> bỏ vì data k có TH đó
- [ ] 8. retrieval score bases on algorithms:
     tf-idf
     bm25: tuning hyperparameter k - saturated index, b- affected document length
- [x] 9. sliding wiki_context sentence. Thay vì kí tự thì xài 1 câu
- [ ] 10. model retrieval combining tf-idf and bm25 is highly considered according to a novel paper

Đọc để hiểu: https://viblo.asia/p/bm25-thuat-toan-xep-hang-cac-van-ban-theo-do-phu-hop-Az45bWGNKxY

In queue

https://medium.com/@papai143/information-retrieval-with-document-re-ranking-with-bert-and-bm25-7c29d738df73

https://www.sciencedirect.com/science/article/pii/S131915781830082X

https://www.sciencedirect.com/science/article/pii/S131915781830082X#b0120

https://en.wikipedia.org/wiki/Question_answering#Types_of_question_answering

https://blog.paperspace.com/how-to-train-question-answering-machine-learning-models/
