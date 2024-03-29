# Telecommunication Company’s Customer Churn Analysis

In the dynamic telecommunications industry, understanding customer behavior and managing churn rates are critical for sustainable growth. In this project, we will explore the association between monthly charges, payment methods, and churn status to develop targeted strategies for reducing churn rates and driving business growth.

Author: [Sally Lee](https://github.com/sallylee0801)

Affiliation: The University of Chicago

The README contains only a brief overview of our goals, investigation process, analysis, and recommendations.

For detailed documentation/analysis/findings, please refer to the [report](https://github.com/sallylee0801/Telecom-Company-Customer-Churn-Analysis/blob/main/Telecommunication%20Company%E2%80%99s%20Customer%20Churn%20Analysis.pdf)

## Business Objective
We aim to leverage the customer churn data to explore the factors influencing customer churn in the telecommunications industry. By examining the association between monthly charges, payment methods, and churn status, the objective is to develop targeted pricing and retention strategies to reduce churn rates, enhance customer satisfaction, and drive sustainable business growth.

## Data Sources
We utilized the Telco-Customer-Churn dataset which comprises 21 columns and 7043 rows. We will focus on key features such as churn status, monthly charges, total charges, tenure, and payment methods.

## EDA
After analysis of data distributions and performing statistics tests, we identified four findings that guided our final investigations:
1. We observed a robust linear and monotonic relationship between monthly charges and average monthly charges, indicating a strong association between the variables.
2. The statistical analysis supports that there's a significant relationship between churn status and preferred payment methods.
3. Churning customers exhibit higher average monthly charges even though the effect size is relatively small.
4. There's a significant difference observed in average monthly charges between churning and non-churning customers.

## Insights: We have identified 4 key features and associations that affect the customer churn status:
1. There's a strong correlation between monthly charges and churn status, emphasizing the need for targeted pricing strategies.

![image](https://github.com/sallylee0801/Telecom-Company-Customer-Churn-Analysis/assets/121594845/6fe5312f-20cd-4d7c-8b89-fc8a6354c59d)

2. The significant relationship between churn status and payment methods highlights the importance of payment method preferences in customer retention.

![image](https://github.com/sallylee0801/Telecom-Company-Customer-Churn-Analysis/assets/121594845/809ec232-f9a2-4a39-a259-0c2caaf11288)

3. The impact of churn status on monthly charges indicates potential opportunities for pricing adjustments.

![image](https://github.com/sallylee0801/Telecom-Company-Customer-Churn-Analysis/assets/121594845/58978a76-dfef-4b03-ba02-ca5071b97627)
![image](https://github.com/sallylee0801/Telecom-Company-Customer-Churn-Analysis/assets/121594845/2bab2470-dd5c-4300-9f35-98973c953074)

4. There's a difference in average monthly charges between churning and non-churning customers, suggesting the need for tailored retention strategies.

## Recommendations
Based on our insights, we have provided several recommendations to reach our intended goal in reducing churn rates and driving business growth:
1. Implement targeted pricing strategies based on customer preferences and behavior.
2. Enhance retention efforts by offering personalized incentives and promotions.
3. Continuously monitor churn rates and adjust strategies accordingly to maintain sustainable business growth.
