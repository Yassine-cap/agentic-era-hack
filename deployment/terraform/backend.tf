terraform {
  backend "gcs" {
    bucket = "qwiklabs-gcp-02-ea851b0ef182-terraform-state"
    prefix = "prod"
  }
}
