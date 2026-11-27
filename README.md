# SmartLogger

スマホの IMU（姿勢・加速度・ジャイロ・地磁気）、気圧、GPS を UDP でリアルタイム送信する Android アプリと、
PC 側で受信してスマホの位置・姿勢・動きを 3D モデルで表示するビューアです。

```
[Android: SmartLogger]  --UDP/JSON (既定 50Hz, port 5005)-->  [PC: pc_viewer/viewer.py (Panda3D)]
```

## 構成

| パス | 内容 |
|---|---|
| `release/SmartLogger.apk` | ビルド済み APK（そのままインストール可能） |
| `android/` | Android アプリのソース（Java、外部ライブラリなし） |
| `pc_viewer/viewer.py` | 3D リアルタイムビューア |
| `pc_viewer/simulate_sender.py` | スマホなしでビューアを試すための疑似送信スクリプト |

## 使い方

### 1. スマホ側

1. `release/SmartLogger.apk` をスマホにインストール（`adb install release/SmartLogger.apk` か、ファイルを転送して開く）。
2. アプリを起動し、位置情報の権限を許可。
3. **PC IP address** に PC の IP アドレス（例 `192.168.1.10`）を入力。`255.255.255.255` のままなら同じ LAN 内にブロードキャストします。
4. ポート（既定 5005）と送信レート（既定 50Hz）を確認して **START**。

スマホと PC は同じネットワーク（Wi-Fi / テザリング）に接続してください。送信中は画面が消えないようになっています（アプリをバックグラウンドにすると送信が止まる端末があります）。

### 2. PC 側

```powershell
pip install -r pc_viewer/requirements.txt
```

```powershell
python pc_viewer/viewer.py
```

初回起動時に Windows ファイアウォールの許可ダイアログが出たら「プライベートネットワーク」を許可してください。

オプション:

| オプション | 既定 | 説明 |
|---|---|---|
| `--port` | 5005 | 受信 UDP ポート |
| `--scale` | 8 | スマホモデルの表示倍率（実寸比） |
| `--alt` | baro | 高さのソース（`baro` 気圧 / `gps`） |
| `--smooth` | 4 | 位置の平滑化レート [1/s] |
| `--accel-scale` | 0.3 | 加速度矢印の長さ [m per m/s²] |

操作: 左ドラッグ = 視点回転 / ホイール = ズーム / `F` = 追従 ON/OFF / `R` = 原点リセット / `T` = 軌跡表示 / `A` = 加速度矢印 / `Esc` = 終了

スマホなしで動作確認:

```powershell
python pc_viewer/simulate_sender.py
```

### 表示内容

- **姿勢**: Android の Rotation Vector センサー（なければ Game Rotation Vector）のクォータニオン
- **水平位置**: GPS の緯度経度を最初の測位点を原点とした東(E)/北(N) [m] に変換
- **高さ**: 気圧から算出した相対高度（気圧センサーがなければ GPS 高度）
- **紫の矢印**: 線形加速度（重力除去済み）をワールド座標に変換したもの
- **オレンジの線**: 移動軌跡
- 座標系: X = 東（赤）, Y = 北（緑）, Z = 上（青）。スマホ上の軸は端末座標（X 右, Y 画面上方向, Z 画面手前）

## UDP パケット形式

1 パケット = UTF-8 の JSON 1 オブジェクト。値が取得できていない項目は省略されます。

```json
{
  "v": 1, "seq": 1234, "t": 1764230000000,
  "q":   [w, x, y, z],
  "acc": [x, y, z],
  "gyr": [x, y, z],
  "mag": [x, y, z],
  "lin": [x, y, z],
  "p":   1013.25,
  "gps": {"lat": 35.0, "lon": 139.0, "alt": 40.0, "acc": 3.0, "spd": 1.2, "brg": 90.0, "prov": "gps", "t": 1764230000000}
}
```

| キー | 単位 | 説明 |
|---|---|---|
| `seq` | - | 連番（パケットロス検出用） |
| `t` | ms | 送信時刻（Unix epoch） |
| `q` | - | 端末座標 → ワールド座標(ENU) の回転クォータニオン |
| `acc` | m/s² | 加速度（重力込み、端末座標） |
| `gyr` | rad/s | 角速度（端末座標） |
| `mag` | µT | 地磁気（端末座標） |
| `lin` | m/s² | 線形加速度（重力除去、端末座標） |
| `p` | hPa | 気圧 |
| `gps.alt` | m | 楕円体高 |
| `gps.acc` | m | 水平精度 |
| `gps.spd` / `gps.brg` | m/s / deg | 速度 / 進行方位 |

## ビルド

Android Studio の JDK と Android SDK (API 36) が必要です。

```powershell
cd android; .\gradlew assembleRelease
```

出力: `android/app/build/outputs/apk/release/app-release.apk`（サイドロード用にデバッグ鍵で署名）。
