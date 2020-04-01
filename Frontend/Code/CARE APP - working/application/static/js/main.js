// A $( document ).ready() block.
$( document ).ready(function() {
    console.log( "ready!" );
});


alert('If you see this alert, then your custom JavaScript script has run!')

$('.colorPicker').on("change", function(){
    //set the value of the input to the value of the select.
        $('.colorDisplay').val($(this).val());
});