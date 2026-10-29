from src.retrieval import retrieve

questions = [
    "Does SMU describe itself as a public or private university?",
    "What is the name of the SMU foundation?",
    "Does SMU Foundation publish policy briefs?",
    "What language courses does LCI list?",
    "Does MedTech list an advising and counseling center?",
    "What exact decision will SMU admissions make about an unnamed applicant?",
]

for question in questions:
    print("\n" + "=" * 70)
    print("QUESTION:", question)

    for i, result in enumerate(retrieve(question, top_k=3), 1):
        print(f"\n[{i}]")
        print("Metadata:", result.get("metadata", {}))
        print("Text:", str(result.get("text", result))[:700])
