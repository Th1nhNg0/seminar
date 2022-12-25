overleaf https://www.overleaf.com/5191338113vqmcbdqjmqwj

📜 TASK:
- [ ] Viết journal cho phần baseline
- [ ] Preprocessing wiki data
- [ ] Metric used for re-rank answer 

  w0 \* retrieval_score + w1 \* predict_score (linear regression)
  
  w0 + w1= 1 
- [ ] Run all dataset for base line: WIKI, UIT, SQUAD, FACEBOOK QA
- [ ] Trường hợp không trả lời được câu hỏi?  -> threshold cho cau tra loi null = 0.6
- [ ] Chạy context của dataset trước để thử nghiệm mấy task trên cho nhanh
- [ ] Context là wiki, chỉ lấy cấu hỏi và gt từ các dataset
