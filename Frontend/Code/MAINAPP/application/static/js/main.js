// A $( document ).ready() block.
$( document ).ready(function() {
    console.log( "ready!" );
});



$(document).ready(function() {
    var url = this.location.pathname;
    var filename = url.substring(url.lastIndexOf('/')+1);
    $('a[href="' + filename + '"]').parent().addClass('active');
});


$(document).ready(function adjustHeight() {
    document.getElementById('filter_box').style.height = document.defaultView.getComputedStyle(document.getElementById('chart_box'), "").getPropertyValue("height");
});

var selector = '.nav li';

$(selector).on('click', function(){
    $(selector).removeClass('active');
    $(this).addClass('active');
});

jQuery(document).ready(function() {
    jQuery(".navbar-toggler").click(function() {
      jQuery(".navbar.navbar-default").slideToggle();
});
});