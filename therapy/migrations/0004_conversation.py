# Generated by Django 4.2.13 on 2024-06-09 10:09

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("therapy", "0003_remove_toneanalysis_features_toneanalysis_emotion_and_more"),
    ]

    operations = [
        migrations.CreateModel(
            name="Conversation",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("session_id", models.CharField(max_length=100)),
                ("history", models.TextField()),
            ],
        ),
    ]
