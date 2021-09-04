# Bank-churn-predict-with-ML
ML 알고리즘을 통한 Bank churn 예측  

outlier 제거, 연속형 변수 scaling(standard scaling)
10 : 1 비율로 train : test set split  

사용 모델  
- logistic regression  
- forward selection  
- ridge, lasso penalty - lambda값은 squared-error, misclassification error를 기준으로 결정  
- scad, mbridge penalty - rmse, log-likelyhood를 기준으로 결정  
- decision tree  
- random forest  
- LDA  
- Adaboosting

평가 기준 - Accuracy
- target 변수의 1이 탈퇴한 회원이므로 탈퇴한 회원을 가장 정확하게 예측하는 모델을 최종모델로 선정

randomization을 통한 performance test
- train, test split 과정부터 모델 학습까지 randomization을 통해 30회 반복 후 평균 Accuracy를 비교
- 평균 Accuracy는 random forest가 가장 좋았으므로 최종 모형은 random forest로 

