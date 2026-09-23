# Original Multimodal RAG corpus

These three PDFs were created exclusively for this project. Aster Devices and all figures are fictional. No documents were copied from another project.

- 01_aster_product_catalog.pdf: product descriptions, pricing/stock/warranty table and playback-duration chart.
- 02_aster_sales_report.pdf: quarterly revenue/orders table and regional order-share chart.
- 03_aster_support_handbook.pdf: return/refund policy, service-target table and support-category chart.

Try:
1. What is the price and warranty of Pulse Max? (USD 219; 24 months)
2. Which model has the longest playback duration? (Pulse Max; 24 hours, figure)
3. What were annual revenue and total orders? (USD 108,000; 660)
4. Which region has the largest order share? (North; 42%, figure)
5. What is the return window and refund processing time? (30 calendar days; 5 business days)
6. What are P1 response and resolution targets? (1 and 8 business hours)
7. Compare Pulse Mini and Pulse Max warranties and summarize the return policy. (12 versus 24 months; 30 calendar days, across documents)

Regenerate with: python scripts/create_project_pdfs.py
Generation dependencies: reportlab and matplotlib. PyMuPDF renders previews into tmp/pdfs for visual checks.
