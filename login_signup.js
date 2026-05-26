function validateForm() {
    var x = document.forms['myForm']['username'].value;
    var y = document.forms['myForm']['password'].value;

    if (x == "" || y == "") {
        alert("Please fill in all fields");
        return false;
    }

    return true;
}

function validateSignupForm() {
    var x = document.forms['signupForm']['username'].value;
    var y = document.forms['signupForm']['email'].value;
    var z = document.forms['signupForm']['password'].value;

    if (x == "" || y == "" || z == "") {
        alert("Please fill in all fields");
        return false;
    }

    return true;
}