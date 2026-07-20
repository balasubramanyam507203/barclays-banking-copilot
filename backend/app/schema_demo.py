from app.models.banking_document import BankingDocument


def main() -> None:
    document = BankingDocument(
        file_name="international_payment_policy.txt",
        document_id="PAY-POL-1042",
        title="International Payment Review Policy",
        version="7.2",
        status="ACTIVE",
        effective_date="2026-04-01",
        department="Payments Compliance",
        region="US",
        classification="Internal Confidential",
        allowed_roles=[
            " payments_analyst ",
            "compliance_analyst",
            "payments_analyst",
        ],
        content=(
            "Enhanced review is required for international "
            "payments involving high-risk destinations."
        ),
    )

    print("Document validation successful.\n")
    print(document.model_dump_json(indent=2))


if __name__ == "__main__":
    main()