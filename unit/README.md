# ✅ unit/ – Testes Unitários

Esta pasta contém os testes automatizados do projeto.  

## 🔧 Executando os testes

```bash
cd unit
pytest -q -m "not slow" --cov=src --cov=api --cov-branch   --cov-report=term-missing:skip-covered --cov-report=xml
```

## 📊 Cobertura de Código

```
Name                                                                              Stmts   Miss Branch BrPart  Cover   Missing
-----------------------------------------------------------------------------------------------------------------------------
/Users/leandro/Documents/Pessoal/pos/fiap_fase5/api/main.py                          20      1      0      0    95%   46
/Users/leandro/Documents/Pessoal/pos/fiap_fase5/api/routes/predict.py                84      8     14      2    90%   70, 108->116, 122-123, 130, 135->139, 137-138, 142-143
/Users/leandro/Documents/Pessoal/pos/fiap_fase5/src/data/feature_engineering.py     213     34     48     13    80%   57-59, 75, 85-86, 88, 102->exit, 115, 150->149, 152->155, 158->163, 164-165, 195, 199, 256-260, 384-403, 406
/Users/leandro/Documents/Pessoal/pos/fiap_fase5/src/data/prepare_data.py             96     18     64     17    77%   23-31, 35, 39->46, 45, 58, 64->68, 69, 75->77, 80, 93, 97->104, 103, 111, 113->115, 115->117, 117->119, 119->121, 126
/Users/leandro/Documents/Pessoal/pos/fiap_fase5/src/models/infer.py                 195     38     32      2    79%   47-48, 264, 319-324, 327-362, 365
/Users/leandro/Documents/Pessoal/pos/fiap_fase5/src/models/train.py                 121     22     14      3    81%   60, 103-104, 132, 209-242, 265
-----------------------------------------------------------------------------------------------------------------------------
TOTAL                                                                               733    121    172     37    81%
```
