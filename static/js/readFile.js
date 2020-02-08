function readURL(input) {
        if (input.files && input.files[0]) {
            var reader = new FileReader();

            reader.onload = function (e) {
                $('#prev_input')
                    .attr('src', e.target.result)
                    .width(300)
                    ;
            };

            reader.readAsDataURL(input.files[0]);
        }
    }

function showResult(divId){
    var x = document.getElementById(divId);
    if (x.style.display == "none"){
        x.style.display = 'block';
    } else {
        x.style.display = "none"
    }

}    