# POLY_SMARTMONEY

## 一键更新（仅干净代码，不涉及私钥）

以下命令只会更新 `v3/v4` 的代码与配置文件，不会改动 `accounts.json`、`state_*.json`、`logs/`。

### 更新 v3（干净代码）

```powershell
$repo='D:/AI/vibe_coding2/POLY_SMARTMONEY'; git -C $repo fetch origin main; git -C $repo checkout origin/main -- copytrade_v3_muti/*.py copytrade_v3_muti/check_*.py copytrade_v3_muti/test_*.py copytrade_v3_muti/copytrade_config.json
```

### 更新 v4（干净代码）

```powershell
$repo='D:/AI/vibe_coding2/POLY_SMARTMONEY'; git -C $repo fetch origin main; git -C $repo checkout origin/main -- copytrade_v4_muti/*.py copytrade_v4_muti/check_*.py copytrade_v4_muti/test_*.py copytrade_v4_muti/copytrade_config.json
```

### 可选：更新后快速自检

```powershell
pytest -q D:/AI/vibe_coding2/POLY_SMARTMONEY/copytrade_v3_muti; pytest -q D:/AI/vibe_coding2/POLY_SMARTMONEY/copytrade_v4_muti
```

## 说明

- 私钥请手动维护在你本地的 `accounts.json` 中。
- 如果你只想更新一个版本，只执行对应那一条命令即可。
