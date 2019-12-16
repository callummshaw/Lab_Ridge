function out=truncname(chaine,term1,term2,choice);

% truncname(chaine,term1,term2,choice);
%TRUNCNAME truncate a string "chaine" 
% before the last  "term" if "choice"='beforelast'
% after  the last  "term" if "choice"='afterlast'
% before the first "term" if "choice"='beforefirst'
% after  the first "term" if "choice"='afterfirst'





locate_term1=findstr(chaine,term1);
locate_term2=findstr(chaine,term2);

if isempty(term2)
    
    if isempty(term1)
        
        out=chaine;
        
    else

        switch choice

            case 'beforelast' % Keep only the strings before the last term1

                out=chaine(1:locate_term1(size(locate_term1,2))-1);

            case 'afterlast' % Keep only the strings after the last term1

                out=chaine(locate_term1(size(locate_term1,2))+1:size(chaine,2));

            case 'beforefirst' % Keep only the strings before the first term1

                out=chaine(1:locate_term1(1)-1);

            case 'afterfirst' % Keep only the strings after the first term1

                out=chaine(locate_term1(1)+1:size(chaine,2));

        end % End switch
        
    end % End if isempty(term1)

    
else 

    if isempty(term1)

        switch choice

            case 'beforelast' % Keep only the strings before the last term2

                out=chaine(1:locate_term2(size(locate_term2,2))-1);

            case 'afterlast' % Keep only the strings after the last term2

                out=chaine(locate_term2(size(locate_term2,2))+1:size(chaine,2));

            case 'beforefirst' % Keep only the strings before the first term2

                out=chaine(1:locate_term2(1)-1);

            case 'afterfirst' % Keep only the strings after the first term2

                out=chaine(locate_term2(1)+1:size(chaine,2));

        end % End switch

        
    else

        out=chaine(locate_term1(size(locate_term1,2))+1:locate_term2(size(locate_term2,2))-1); % Keep only the strings beetween term1 and term2

    end   % End if isempty(term1)


end % End if isempty(term2)



