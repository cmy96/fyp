// autofill functions for form input

function getT() {

        var size_precise = $('#size_precise').val();
        if (size_precise ==  0 ) {
            $('#tstage').val('t0')
        } else if (size_precise <= 0.1) {
            $('#tstage').val('t1mic')
        } else if (size_precise >0.1 && size_precise <=0.5) {
            $('#tstage').val('t1a')
        } else if (size_precise > 0.5 && size_precise <=1) {
            $('#tstage').val('t1b')
        } else if (size_precise > 1 && size_precise <=2) {
            $('#tstage').val('t1c')
        } else if (size_precise > 2 && size_precise <=5 ) {
            $('#tstage').val('t2')
        } else if(size_precise > 5) {
            $('#tstage').val('t3')
        } else {
            $('#tstage').val('')
        }
    }

function getAge() {
    $('#age').val(123)
}

function getN() {
    var nodes_pos = $('#nodespos').val();
    var size_precise = $('#size_precise').val();
    if (nodes_pos ==0) {
        $('#nstage').val('n0')
    } else if (nodes_pos >=1 && nodes_pos <=3 && size_precise < 2) {
        $('#nstage').val('n1')
    } else if (nodes_pos >=1 && nodes_pos <=3 && size_precise >=2) {
        $('#nstage').val('n1')
    } else if (nodes_pos >= 4 && nodes_pos <= 9) {
        $('#nstage').val('n2')
    } else if (nodes_pos >=10 ) {
        $('#nstage').val('n3')
    }
}


function getStage() {
    var t_stage = $('#tstage').val();
    var n_stage = $('#nstage').val();
    var m_stage = $('#mStage').val();

    if (t_stage =='tis' && n_stage =='n0' && m_stage =='m0') {
        $('#stage').val('dcis/lcis non-invasive')
    } else if (t_stage.indexOf("t1") > -1 && n_stage =='n0' && m_stage =='m0') {
        $('#stage').val('stage 1a')
    } else if (t_stage =='t0' && n_stage=='n1mic' && m_stage=='m0'){
        $('#stage').val('stage 1b')
    } else if (t_stage.indexOf("t1") > -1 && n_stage=='n1mic' && m_stage=='m0'){
        $('#stage').val('stage 1b')
    } else if (t_stage =='t0' && n_stage.indexOf("n1") > -1 && m_stage=='m0') {
        $('#stage').val('stage 2a')
    } else if (t_stage.indexOf("t1") > -1 && n_stage.indexOf("n1") > -1 && m_stage=='m0') {
        $('#stage').val('stage 2a')
    } else if (t_stage =='t2' && n_stage=='n0' && m_stage=='m0') {
        $('#stage').val('stage 2a')
    } else if (t_stage =='t2' && n_stage=='n1' && m_stage=='m0') {
        $('#stage').val('stage 2b')
    } else if (t_stage =='t3' && n_stage=='n0' && m_stage=='m0') {
        $('#stage').val('stage 2b')
    } else if (t_stage =='t0' && n_stage=='n2' && m_stage=='m0') {
        $('#stage').val('stage 3a')
    } else if (t_stage.indexOf("t1") > -1 && n_stage=='n2' && m_stage=='m0') {
        $('#stage').val('stage 3a')
    } else if (t_stage =='t2' && n_stage=='n2' && m_stage=='m0') {
        $('#stage').val('stage 3a')
    } else if (t_stage =='t3' && n_stage=='n1' && m_stage=='m0') {
        $('#stage').val('stage 3a')
    } else if (t_stage =='t3' && n_stage=='n2' && m_stage=='m0') {
        $('#stage').val('stage 3a')
    } else if (t_stage =='t4' && n_stage=='n0' && m_stage=='m0'){
        $('#stage').val('stage 3b')
    } else if (t_stage =='t4' && n_stage=='n1' && m_stage=='m0'){
        $('#stage').val('stage 3b')
    } else if (t_stage =='t4' && n_stage=='n2' && m_stage=='m0'){
        $('#stage').val('stage 3b')
    } else if ( n_stage=='n3' && m_stage=='m0'){
        $('#stage').val('stage 3c')
    } else if (m_stage=='m1'){
        $('#stage').val('stage 4')
    } 


}

function t_prompt() {
    var size_precise = $('#size_precise').val();
    var t_stage = $('#tstage').val();

    if (size_precise.length === 0) {
        $('#tstage').css('border-color', '')
    } else if (size_precise == 0 && t_stage!='t0') {
        $('#tstage').css('border-color', 'red')
    } else if (size_precise <= 0.1 && t_stage!='t1mic') {
        $('#tstage').css('border-color', 'red')
    } else if (size_precise >0.1 && size_precise <=0.5 && t_stage!='t1a') {
        $('#tstage').css('border-color', 'red')
    } else if (size_precise > 0.5 && size_precise <=1 && t_stage!='t1b') {
        $('#tstage').css('border-color', 'red')
    } else if (size_precise > 1 && size_precise <=2 && t_stage!='t1c') {
        $('#tstage').css('border-color', 'red')
    } else if (size_precise > 2 && size_precise <=5 && t_stage!='t2') {
        $('#tstage').css('border-color', 'red')
    } else if (size_precise > 5 && t_stage!='t3') {
        $('#tstage').css('border-color', 'red')
    } else {
        $('#tstage').css('border-color', '')
    }

}

function n_prompt() {
    var nodes_pos = $('#nodespos').val();
    var size_precise = $('#size_precise').val();
    var n_stage = $('#nstage').val();

    if (nodes_pos.length === 0) {
        $('#nstage').css('border-color', '')
    } else if (nodes_pos >=1 && nodes_pos <=3 && size_precise < 2 && n_stage!='n1') {
        $('#nstage').css('border-color', 'red')
    } else if (nodes_pos >=1 && nodes_pos <=3 && size_precise >=2 && n_stage!='n1a') {
        $('#nstage').css('border-color', 'red')
    } else if (nodes_pos >= 4 && nodes_pos <= 9 && n_stage !='n2') {
        $('#nstage').css('border-color', 'red')
    } else if (nodes_pos >=10 && n_stage!='n3') {
        $('#nstage').css('border-color', 'red')
    } else {
        $('#nstage').css('border-color', '')
    }
}

function stage_prompt() {
    var t_stage = $('#tstage').val();
    var n_stage = $('#nstage').val();
    var m_stage = $('#mStage').val();
    var stage = $('#stage').val();

    if (t_stage =='tis' && n_stage =='n0' && m_stage =='m0' && stage!='dcis/lcis non-invasive') {
        $('#stage').css('border-color', 'red')
    } else if (t_stage.indexOf("t1") > -1 && n_stage =='n0' && m_stage =='m0' && stage!='stage 1a') {
        $('#stage').css('border-color', 'red')
    } else if (t_stage =='t0' && n_stage=='n1mic' && m_stage=='m0' && stage!='stage 1b'){
        $('#stage').css('border-color', 'red')
    } else if (t_stage.indexOf("t1") > -1 && n_stage=='n1mic' && m_stage=='m0' && stage!='stage 1b'){
        $('#stage').css('border-color', 'red')
    } else if (t_stage =='t0' && n_stage.indexOf("n1") > -1 && m_stage=='m0' && stage!='stage 2a') {
        $('#stage').css('border-color', 'red')
    } else if (t_stage =='t1' && n_stage=='n1' && m_stage=='m0' && stage!='stage 2a') {
        $('#stage').css('border-color', 'red')
    } else if (t_stage =='t2' && n_stage=='n0' && m_stage=='m0' && stage!='stage 2a') {
        $('#stage').css('border-color', 'red')
    } else if (t_stage =='t2' && n_stage=='n1' && m_stage=='m0' && stage!='stage 2b') {
        $('#stage').css('border-color', 'red')
    } else if (t_stage =='t3' && n_stage=='n0' && m_stage=='m0' && stage!='stage 2b') {
        $('#stage').css('border-color', 'red')
    } else if (t_stage =='t0' && n_stage=='n2' && m_stage=='m0' && stage!='stage 3a') {
        $('#stage').css('border-color', 'red')
    } else if (t_stage.indexOf("t1") > -1 && n_stage=='n2' && m_stage=='m0' && stage !='stage 3a') {
        $('#stage').css('border-color', 'red')
    } else if (t_stage =='t2' && n_stage=='n2' && m_stage=='m0' && stage != 'stage 3a') {
        $('#stage').css('border-color', 'red')
    } else if (t_stage =='t3' && n_stage=='n1' && m_stage=='m0' && stage != 'stage 3a') {
        $('#stage').css('border-color', 'red')
    } else if (t_stage =='t3' && n_stage=='n2' && m_stage=='m0' && stage != 'stage 3a') {
        $('#stage').css('border-color', 'red')
    } else if (t_stage =='t4' && n_stage=='n0' && m_stage=='m0' && stage!='stage 3b'){
        $('#stage').css('border-color', 'red')
    } else if (t_stage =='t4' && n_stage=='n1' && m_stage=='m0' && stage!='stage 3b'){
        $('#stage').css('border-color', 'red')
    } else if (t_stage =='t4' && n_stage=='n2' && m_stage=='m0' && stage!='stage 3b'){
        $('#stage').css('border-color', 'red')
    } else if ( n_stage=='n3' && m_stage=='m0' && stage!='stage 3c'){
        $('#stage').css('border-color', 'red')
    } else if (m_stage=='m1' && stage!='stage 4'){
        $('#stage').css('border-color', 'red')
    } else {
        $('#stage').css('border-color', '')
    }
}




function navbar() {
    "use strict";


    $('ul.navbar-nav > li').click(function(e) {
        e.preventDefault();
        $('ul.navbar-nav > li').removeClass('active');
        $(this).addClass('active');
    });
}


function loader() {
    var nodes_pos = $('#nodespos').val();
    var size_precise = $('#size_precise').val();    
    var t_stage = $('#tstage').val();
    var n_stage = $('#nstage').val();
    var m_stage = $('#mStage').val();
    var stage = $('#stage').val();
    var diff_grade = $('#diff').val();
    var er = $('#ER').val();
    var pr = $('#PR').val();
    var her2 = $('#Her2').val();

    if (nodes_pos.length !== 0 && size_precise.length!==0 && t_stage.length!==0 && n_stage.length!==0 && m_stage.length!==0
        && stage.length!==0 && diff_grade.length!==0 && er.length!==0 && pr.length!==0 && her2.length!==0) {
            

            $('body').append('<div style="" id="loadingDiv_index"><div class="loader_index"></div></div>');
            $(window).on('load', function(){
            setTimeout(removeLoader, 2000); //wait for page load PLUS two seconds.
            });
            function removeLoader(){
                $( "#loadingDiv_index" ).fadeOut(500, function() {
                // fadeOut complete. Remove the loading div
                $( "#loadingDiv_index" ).hide(); //makes page more lightweight 
            });  
            }
        }
    }


