$( document ).ready(function() {
    console.log( "ready!" );

    $("#myform").on("submit", function(){
        $("#pageloader").fadeIn();
      });

});

$('form').on('click', 'button:not([type="submit"])', function(e){
    e.preventDefault();
});



// $(document).ready(function(){
//     //when the select changes:
// $('.colorPicker').on("change", function(){
//     //set the value of the input to the value of the select.
//         $('.colorDisplay').val($(this).val());
// });
// });

// $('.colorPicker').on("change", function(){
//     //set the value of the input to the value of the select.
//         $('.colorDisplay').val($(this).val());
// });



