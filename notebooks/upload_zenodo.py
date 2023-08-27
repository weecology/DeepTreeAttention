import zenodopy

# always start by creating a Client object
zeno = zenodopy.Client()

# list project id's associated to zenodo account
zeno.list_projects

# create a project
zeno.create_project(title="test_project", upload_type="other")
# your zeno object now points to this newly created project

# create a file to upload
with open("~/test_file.txt", "w+") as f:
    f.write("Hello from zenodopy")

# upload file to zenodo
zeno.upload_file("~/test.file.txt")