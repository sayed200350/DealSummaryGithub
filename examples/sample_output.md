# Deal Summary & QA Report

Generated 3 analysis results.

## Deal 1 (CSV Row 2)

**Original Text:**
```
Amazing deal on Sony WH-1000XM4 headphones! Get 50% off the regular price of €199.99. Premium noise cancellation technology perfect for work and travel. Limited time offer - grab yours at https://example-store.com/sony-headphones today!
```

**Generated Summary:**
> Incredible Sony WH-1000XM4 headphones deal! Save 50% off €199.99 with premium noise cancellation. Perfect for work & travel. Limited time - shop now at https://example-store.com/sony-headphones!

*Length: 184 characters*

**Quality Assurance Results:**
```json
{
  "has_price": true,
  "price_value": 199.99,
  "has_source_url": true,
  "source_url": "https://example-store.com/sony-headphones",
  "within_length_limit": true,
  "missing_keywords": [],
  "duplicate_suspect": false,
  "notes": "Excellent deal content with all required elements present."
}
```

---

## Deal 2 (CSV Row 3)

**Original Text:**
```
Free shipping on all Nike Air Max sneakers this weekend! Comfortable running shoes with advanced cushioning starting at €89.99. Perfect for athletes and casual wear. Visit our store or order online at https://sneaker-shop.com/nike-deal
```

**Generated Summary:**
> Free shipping on Nike Air Max sneakers this weekend! Starting at €89.99 with advanced cushioning. Perfect for athletes & casual wear. Order at https://sneaker-shop.com/nike-deal!

*Length: 164 characters*

**Quality Assurance Results:**
```json
{
  "has_price": true,
  "price_value": 89.99,
  "has_source_url": true,
  "source_url": "https://sneaker-shop.com/nike-deal",
  "within_length_limit": true,
  "missing_keywords": [
    "deal"
  ],
  "duplicate_suspect": false,
  "notes": "Good content with price and URL, but missing 'deal' keyword in the text."
}
```

---

## Deal 3 (CSV Row 4)

**Original Text:**
```
MacBook Pro 14-inch now available with 25% student discount! Powerful M2 chip, 16GB RAM, 512GB SSD for just €1899. Ideal for creative professionals and students. Apply at https://education-store.com/macbook-deal with valid student ID.
```

**Generated Summary:**
> MacBook Pro 14-inch with 25% student discount! M2 chip, 16GB RAM, 512GB SSD for €1899. Perfect for creatives & students. Apply at https://education-store.com/macbook-deal!

*Length: 164 characters*

**Quality Assurance Results:**
```json
{
  "has_price": true,
  "price_value": 1899.0,
  "has_source_url": true,
  "source_url": "https://education-store.com/macbook-deal",
  "within_length_limit": true,
  "missing_keywords": [
    "deal"
  ],
  "duplicate_suspect": false,
  "notes": "High-value deal with all elements present except 'deal' keyword in main text."
}
```

---
