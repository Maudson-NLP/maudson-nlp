$(function(){
   // Bind advanced options click listener
    $('#show-advanced').on('click', function(){
       $('#advanced-options').toggle();
    });
});

var result_id;
var intervalId;
var attempts;

function doUpload(){
    attempts = 0;
    $('#summarization-result').text('Summarizing...');
    var formData = new FormData();

    var textToSummarize = $('#textToSummarize').val();
    if (!textToSummarize) {
        var fileInput = document.getElementById('the-file');
        var file = fileInput.files[0];
        formData.append('file', file);
    } else {
        formData.append('textToSummarize', textToSummarize);
    }

    colnames = $('#the-column-names').val();
    formData.append('columns', colnames);
    formData.append('l-value', $('#l-value').val());
    formData.append('ngram-min', $("#ngram-min").val());
    formData.append('ngram-max', $("#ngram-max").val());
    formData.append('tfidf', $("#tfidf").prop("checked"));
    formData.append('use-svd', $("#svd").prop("checked"));
    formData.append('scale-vectors', $("#scale-vectors").prop("checked"));
    formData.append('top-k', $("#top-k").val());
    formData.append('use-noun-phrases', $("#noun-phrases").prop("checked"));
    formData.append('split-longer-sentences', $("#split-longer-sentences").prop("checked"));
    formData.append('to-split-length', $("#to-split-length").val());
    formData.append('group-by', $("#group-by").val());
    // Local Only Features
    // formData.append('extract-sibling-sents', $("#extract-sibling-sents").prop("checked"));
    // formData.append('exclude-misspelled', $("#exclude-misspelled").prop("checked"));
    formData.append('extract-sibling-sents', false);
    formData.append('exclude-misspelled', false);

    var xhr = new XMLHttpRequest();
    xhr.onreadystatechange = function() {
        if (xhr.readyState == XMLHttpRequest.DONE) {

            if (xhr.status === 200) {
                result_id = xhr.responseText;
                intervalId = setInterval(checkResults, 10000);
            } else {
                $('body').empty().append(xhr.responseText);
                console.log("Error", xhr.statusText);
                clearInterval(intervalId);
            }

        }
    };

    xhr.open('POST', '/summarize', true);
    xhr.send(formData);
}



function checkResults() {
    attempts += 1;

    var fd = new FormData();
    fd.append('result_id', result_id);

    var xhr = new XMLHttpRequest();
    xhr.onreadystatechange = function() {
        if (xhr.readyState == XMLHttpRequest.DONE) {

            if (xhr.status === 200) {
                var json = JSON.parse(xhr.responseText);
                // Join the array of strings by simple spaces - todo
                json.map(function(j){ j[1] = j[1].join(' '); return j });

                var $compiled = $('#summarization--template').tmpl({
                    summaries: json
                });
                $('#summarization-result').empty().append($compiled);

                clearInterval(intervalId);

            } else if (xhr.status == 404) {
                if (attempts >= 25) {
                    $('#summarization-result').text('Taking longer than expected - possible memory utilization error.');
                } else {
                    $('#summarization-result').text('Please wait...');
                }
            } else {
                $('body').empty().append(xhr.responseText);
                console.log("Error", xhr.statusText);
                clearInterval(intervalId);
            }
        }
    };

    xhr.open('POST', '/summary_result', true);
    xhr.send(fd);

}