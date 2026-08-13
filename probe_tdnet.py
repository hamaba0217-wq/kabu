# -*- coding: utf-8 -*-
"""
TDnet適時開示アドオンが、あなたのJ-Quants契約で使えるか疎通確認する。

TDnet適時開示は2026年5月開始の別料金アドオン。標準プランには含まれない。
実際にAPIを叩いて、取れるか・取れないか（契約の有無）を事実で確認する。
"""

from __future__ import annotations


def main():
    from sources import JQuants
    jq = JQuants()
    print("=" * 72)
    print("TDnet適時開示 / EDINET大株主 アドオンの疎通確認")
    print("=" * 72)
    print("実際にAPIを叩いて、あなたの契約で取得できるか確認します...\n")

    # --- TDnet適時開示 ---
    print("─" * 72)
    print("① TDnet適時開示（あなたの観点：ニュース・材料）")
    print("─" * 72)
    _report(jq.probe_tdnet(), "TDnet適時開示")

    # --- EDINET大株主 ---
    print("\n" + "─" * 72)
    print("② EDINET大株主（あなたの観点：資金の流れ・大株主の動き）")
    print("─" * 72)
    _report(jq.probe_edinet(), "EDINET大株主")

    print("\n" + "=" * 72)
    print("補足（事実）:")
    print("  どちらも別料金アドオンの可能性があります（標準プランに含まれない）。")
    print("  『plan』『403』が出たら未契約。契約状況は J-Quants マイページで確認可。")
    print("  取れたデータがあれば、それを使って5倍株の前兆分析に組み込めます。")


def _report(result, label):
    status = result.get("status")
    if status == "ok":
        print(f"✅ {label}：取得できました！ 契約に含まれています。")
        print(f"   メソッド: {result['method']}（{result.get('how','')}）")
        print(f"   取得行数: {result['rows']}")
        print(f"   列: {result['columns']}")
    elif status == "empty_or_error":
        detail = result.get("detail", "")
        print(f"△ {label}：メソッドは動きましたが、データが返りませんでした。")
        print(f"   詳細: {detail}")
        print(f"   → 契約に含まれず空を返すか、引数指定が必要な可能性。")
        print(f"     403は出ていないので、契約がある可能性も残ります。")
    elif status == "error":
        detail = result.get("detail", "")
        print(f"△ {label}：メソッドは動いたが、取得でエラーになりました。")
        print(f"   エラー: {detail}")
        if "plan" in detail.lower() or "403" in detail or "not available" in detail.lower():
            print(f"   → あなたのプランには含まれていません（別料金アドオン）。")
        else:
            print(f"   → 一時的な通信不良か、引数の形が違う可能性があります。")
    elif status == "no_method":
        print(f"△ {label}：ライブラリにメソッドがありません。")
        print(f"   {result.get('detail','')}")
    else:
        print(f"{label}：不明な結果 {result}")


def _old_main():
    from sources import JQuants
    jq = JQuants()
    print("=" * 72)
    print("TDnet適時開示アドオンの疎通確認")
    print("=" * 72)
    print("実際にAPIを叩いて、あなたの契約で取得できるか確認します...\n")

    result = jq.probe_tdnet()
    status = result.get("status")

    if status == "ok":
        print("✅ 取得できました！ TDnetアドオンは契約に含まれています。")
        print(f"   使用メソッド: {result['method']}（{result.get('how','')}）")
        print(f"   取得行数: {result['rows']}")
        print(f"   列: {result['columns']}")
        print("\n→ 5倍株の『材料ニュースの有無』を、開示件数で分析できます。")
        print("  次のステップとして、開示データを前兆分析に組み込めます。")
        print("  『どの列が開示の種類か』が分かれば、材料の中身も分類できます。")
    elif status == "error":
        print("△ メソッドは存在しますが、取得でエラーになりました。")
        print(f"   メソッド: {result['method']}")
        print(f"   エラー: {result['detail']}")
        print("\n→ エラー内容が『403』『Forbidden』『権限』『plan』等なら、")
        print("  TDnetアドオン未契約の可能性が高いです（別料金アドオン）。")
        print("  それ以外のエラーなら、一時的な通信不良かもしれません。")
    elif status == "no_method":
        print("△ ライブラリにTDnet取得メソッドが見つかりませんでした。")
        print(f"   {result['detail']}")
        print("\n→ jquantsapi を最新版に更新すると追加される可能性があります：")
        print("     py -m pip install -U jquants-api-client")
        print("  更新後、もう一度この確認を実行してください。")
    else:
        print(f"不明な結果: {result}")

    print("\n" + "=" * 72)
    print("補足（事実）:")
    print("  TDnet適時開示は2026年5月18日開始の別料金アドオンです。")
    print("  標準プランには含まれないため、未契約なら追加契約が必要です。")
    print("  契約状況は J-Quants のマイページでも確認できます。")


if __name__ == "__main__":
    main()
