overleaf https://www.overleaf.com/5191338113vqmcbdqjmqwj

🏃‍♂️ RESULT:

- BASE LINE: [baseline summary](baseline_summary.md)

📜 TO-DO:

- [ ] 1. Viết journal cho phần baseline
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
